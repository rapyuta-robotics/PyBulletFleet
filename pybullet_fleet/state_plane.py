"""Shared-memory state plane: the simulation writes the fleet's state each step; other processes read it.

A ROS bridge, a viewer or a recorder that wants every robot's pose each step
used to run inside the simulation process and pay for it on the stepping
thread: at 1000 robots, building one ``FleetState`` took about 9 ms. The state
plane moves that work out of the process. The simulation writes a snapshot of
the fleet (one row per agent: pose, velocity, moving flag, battery) into a ring
of slots in a ``multiprocessing.shared_memory`` segment, and any number of
readers in other processes copy the slot they need. No ROS is involved here;
``ros2_bridge`` publishes from a reader.

Layout of the segment, little-endian, all offsets fixed by :class:`StateLayout`::

    header      magic, version, slots, capacity, name width, reader slots, timestep,
                head (seq of the last committed slot), generation, mode,
                reader table (active, tail, heartbeat) x reader slots
    name table  capacity x name width bytes; rewritten when the generation changes
    slot[i]     seq_begin, seq_end, sim_time, step, generation, n_rows, then the
                row arrays: object_id, pos, orn, lin_vel, ang_vel, flags, battery

Each slot is a seqlock: the writer stores ``seq_begin``, the rows, then
``seq_end``; a reader that finds the two unequal has read a slot being written
and reads again. The handoff that wakes a reader, and the backpressure that
makes the writer wait in ``every_step`` mode, go through named pipes (one per
reader, one for the writer): a blocking read on a pipe is an OS wait, not a
busy loop, and the pipe syscalls order the shared-memory stores before the
wakeup, which Python's numpy stores alone do not guarantee on every CPU.

Delivery modes:

- ``every_step``: every snapshot reaches every registered reader. When the
  ring is full the writer waits for the slowest reader, up to ``wait_timeout``
  seconds; past that it logs a warning once and overwrites the oldest slot
  until the readers catch up.
- ``latest``: the writer never waits; a reader takes the newest committed slot
  and skips the rest.

The writer owns the segment and the pipes and removes them on :meth:`close`.
A reader only attaches, and tells Python's resource tracker not to unlink the
segment when it exits (python/cpython#82300).
"""

from __future__ import annotations

import errno
import logging
import math
import os
import select
import tempfile
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Any, Iterable, Optional, Sequence

import numpy as np

# Only warnings and rare notices are logged here; nothing on the per-step path.
_warn_logger = logging.getLogger(__name__)

MAGIC = 0x50424653  # "PBFS"
LAYOUT_VERSION = 1
MODE_EVERY_STEP = "every_step"
MODE_LATEST = "latest"
MODES = (MODE_EVERY_STEP, MODE_LATEST)
DEFAULT_SLOTS = 8
DEFAULT_READER_SLOTS = 8
DEFAULT_NAME_WIDTH = 64
DEFAULT_WAIT_TIMEOUT_S = 1.0
HEADER_BYTES = 4096

FLAG_MOVING = 1
FLAG_CHARGING = 2
FLAG_HAS_BATTERY = 4

_HEADER_DTYPE = np.dtype(
    [
        ("magic", "<u4"),
        ("version", "<u4"),
        ("slots", "<u4"),
        ("capacity", "<u4"),
        ("name_width", "<u4"),
        ("reader_slots", "<u4"),
        ("timestep", "<f8"),
        ("head", "<i8"),
        ("generation", "<i8"),
        ("mode", "<u4"),
        ("pad", "<u4"),
        ("wait_timeout", "<f8"),
    ]
)
# Writers created in this process: a reader here must not unregister their segment.
_owned_names: set[str] = set()
_READER_DTYPE = np.dtype([("active", "<u4"), ("pad", "<u4"), ("tail", "<i8"), ("heartbeat", "<f8")])
_SLOT_META_DTYPE = np.dtype(
    [
        ("seq_begin", "<i8"),
        ("seq_end", "<i8"),
        ("sim_time", "<f8"),
        ("step", "<i8"),
        ("generation", "<i8"),
        ("n_rows", "<i8"),
    ]
)


def _row_dtype(capacity: int) -> np.dtype:
    return np.dtype(
        [
            ("object_id", "<i8", (capacity,)),
            ("pos", "<f8", (capacity, 3)),
            ("orn", "<f8", (capacity, 4)),
            ("lin_vel", "<f8", (capacity, 3)),
            ("ang_vel", "<f8", (capacity, 3)),
            ("flags", "u1", (capacity,)),
            ("battery", "<f4", (capacity,)),
        ]
    )


@dataclass(frozen=True)
class StateLayout:
    """Where everything sits in the segment, derived from the header fields."""

    slots: int
    capacity: int
    name_width: int
    reader_slots: int

    @property
    def reader_table_offset(self) -> int:
        return _HEADER_DTYPE.itemsize

    @property
    def name_table_offset(self) -> int:
        return HEADER_BYTES

    @property
    def name_table_bytes(self) -> int:
        return self.capacity * self.name_width

    @property
    def slots_offset(self) -> int:
        return self.name_table_offset + self.name_table_bytes

    @property
    def rows_dtype(self) -> np.dtype:
        return _row_dtype(self.capacity)

    @property
    def slot_bytes(self) -> int:
        return _SLOT_META_DTYPE.itemsize + self.rows_dtype.itemsize

    @property
    def total_bytes(self) -> int:
        return self.slots_offset + self.slots * self.slot_bytes

    def slot_offset(self, index: int) -> int:
        return self.slots_offset + index * self.slot_bytes

    def __post_init__(self) -> None:
        if self.slots < 2:
            raise ValueError("a state plane needs at least 2 slots")
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1 row")
        if self.reader_slots < 1 or _HEADER_DTYPE.itemsize + self.reader_slots * _READER_DTYPE.itemsize > HEADER_BYTES:
            raise ValueError(
                f"reader_slots must be between 1 and {(HEADER_BYTES - _HEADER_DTYPE.itemsize) // _READER_DTYPE.itemsize}"
            )


@dataclass(frozen=True)
class FleetFrame:
    """One step of fleet state, as arrays with one row per agent."""

    sim_time: float
    step: int
    names: tuple[str, ...]
    object_id: np.ndarray  # (N,) int64
    pos: np.ndarray  # (N, 3)
    orn: np.ndarray  # (N, 4) xyzw
    lin_vel: np.ndarray  # (N, 3)
    ang_vel: np.ndarray  # (N, 3)
    is_moving: np.ndarray  # (N,) bool
    battery_soc: np.ndarray  # (N,) float32, NaN when the agent has no battery
    is_charging: np.ndarray  # (N,) bool

    @property
    def n_rows(self) -> int:
        return int(self.object_id.shape[0])


@dataclass(frozen=True)
class Snapshot(FleetFrame):
    """A frame read back from the plane, with the slot's sequence number and generation."""

    seq: int = -1
    generation: int = -1


def collect_fleet_frame(sim_core: Any, agents: Optional[Sequence[Any]] = None) -> FleetFrame:
    """The current state of *sim_core*'s agents as a :class:`FleetFrame`.

    With a simulation core (``agents`` not given) the frame is a slice of the
    core's columns: poses, velocities and the moving flags come out as
    arrays, and only the agents with a battery plugin are read one by one.
    The agents' rows and names are cached until the fleet changes (an object
    added, removed or renamed). With an explicit ``agents`` sequence, or a
    core without columns, the agents are walked once per call.
    """
    store = getattr(sim_core, "_store", None)
    if agents is None and store is not None and hasattr(sim_core, "_fleet_revision"):
        return _collect_from_columns(sim_core, store)
    if agents is None:
        agents = tuple(getattr(sim_core, "agents", ()))
    n = len(agents)
    poses = [agent.get_pose() for agent in agents]
    object_id = np.fromiter((int(getattr(agent, "object_id", -1)) for agent in agents), dtype=np.int64, count=n)
    # The same fallback as FleetStateProvider, so both name a robot alike.
    names = tuple(str(getattr(agent, "name", None) or f"agent_{oid}") for agent, oid in zip(agents, object_id))
    pos = np.array([pose.position for pose in poses], dtype=float).reshape(n, 3)
    orn = np.array([pose.orientation for pose in poses], dtype=float).reshape(n, 4)
    lin_vel = np.zeros((n, 3))
    velocities = [getattr(agent, "velocity", None) for agent in agents]
    try:
        if velocities and all(v is not None for v in velocities):
            lin_vel[:] = np.asarray(velocities, dtype=float).reshape(n, -1)[:, :3]
        else:
            raise ValueError
    except ValueError:  # a velocity missing or of another length: row by row
        for i, velocity in enumerate(velocities):
            if velocity is not None:
                lin_vel[i] = np.asarray(velocity, dtype=float)[:3]
    ang_vel = np.zeros((n, 3))
    for i, angular in enumerate(getattr(agent, "angular_velocity", None) for agent in agents):
        if angular is None:
            continue
        if isinstance(angular, (int, float)) or np.ndim(angular) == 0:
            ang_vel[i, 2] = float(angular)
        else:
            ang_vel[i] = np.asarray(angular, dtype=float)[:3]
    is_moving = np.fromiter((bool(getattr(agent, "is_moving", False)) for agent in agents), dtype=bool, count=n)
    socs = [getattr(agent, "battery_soc", None) for agent in agents]
    battery = np.array([np.nan if soc is None else soc for soc in socs], dtype=np.float32).reshape(n)
    is_charging = np.fromiter(
        (soc is not None and bool(getattr(agent, "is_charging", False)) for agent, soc in zip(agents, socs)),
        dtype=bool,
        count=n,
    )
    return FleetFrame(
        sim_time=float(getattr(sim_core, "sim_time", 0.0)),
        step=int(getattr(sim_core, "step_count", 0)),
        names=names,
        object_id=object_id,
        pos=pos,
        orn=orn,
        lin_vel=lin_vel,
        ang_vel=ang_vel,
        is_moving=is_moving,
        battery_soc=battery,
        is_charging=is_charging,
    )


def _collect_from_columns(sim_core: Any, store: Any) -> FleetFrame:
    """A :class:`FleetFrame` sliced from the core's columns (see :func:`collect_fleet_frame`)."""
    revision = sim_core._fleet_revision
    cache = getattr(sim_core, "_fleet_frame_cache", None)
    if cache is None or cache["revision"] != revision:
        agents = tuple(sim_core.agents)
        n = len(agents)
        object_id = np.fromiter((int(agent.object_id) for agent in agents), dtype=np.int64, count=n)
        # The same fallback as FleetStateProvider, so both name a robot alike.
        names = tuple(str(getattr(agent, "name", None) or f"agent_{oid}") for agent, oid in zip(agents, object_id))
        batteries = []
        for i, agent in enumerate(agents):
            plugin = getattr(agent, "battery_plugin", None)
            if plugin is not None:
                batteries.append((i, plugin))
        cache = {
            "revision": revision,
            "object_id": object_id,
            "names": names,
            "rows": store.rows(object_id.tolist()),
            "batteries": batteries,
        }
        sim_core._fleet_frame_cache = cache
    rows = cache["rows"]
    n = len(rows)
    ang_vel = np.zeros((n, 3))
    ang_vel[:, 2] = store.ang_vel[rows]
    battery = np.ones(n, dtype=np.float32)
    is_charging = np.zeros(n, dtype=bool)
    for i, plugin in cache["batteries"]:
        battery[i] = plugin.soc
        is_charging[i] = bool(plugin.is_charging)
    return FleetFrame(
        sim_time=float(sim_core.sim_time),
        step=int(sim_core.step_count),
        names=cache["names"],
        object_id=cache["object_id"],
        pos=store.pos[rows].copy(),
        orn=store.orn[rows].copy(),
        lin_vel=store.lin_vel[rows].copy(),
        ang_vel=ang_vel,
        is_moving=store.is_moving[rows].copy(),
        battery_soc=battery,
        is_charging=is_charging,
    )


# ---------------------------------------------------------------------------
# Named pipes: the wakeup and backpressure channel
# ---------------------------------------------------------------------------


def _pipe_dir() -> str:
    return os.environ.get("PYBULLET_FLEET_STATE_PLANE_DIR", tempfile.gettempdir())


def _pipe_path(name: str, who: str) -> str:
    return os.path.join(_pipe_dir(), f"pbf_state_{name}.{who}.fifo")


def _make_fifo(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    os.mkfifo(path, 0o600)


def _open_fifo_for_reading(path: str) -> int:
    return os.open(path, os.O_RDONLY | os.O_NONBLOCK)


def _open_fifo_for_writing(path: str) -> Optional[int]:
    """A non-blocking write end, or None while nobody holds the read end open."""
    try:
        return os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno in (errno.ENXIO, errno.ENOENT):
            return None
        raise


def _poke(fd: Optional[int]) -> None:
    """One byte into a pipe; a full pipe means the other side is far behind and already awake."""
    if fd is None:
        return
    try:
        os.write(fd, b"\x01")
    except BlockingIOError:
        pass
    except BrokenPipeError:
        pass


def _drain(fd: int) -> None:
    try:
        while os.read(fd, 4096):
            pass
    except BlockingIOError:
        pass


def _wait(fd: int, timeout: Optional[float]) -> bool:
    """Block until the pipe has a byte (True) or *timeout* seconds pass (False)."""
    ready, _, _ = select.select([fd], [], [], timeout)
    if ready:
        _drain(fd)
        return True
    return False


# ---------------------------------------------------------------------------
# Writer (simulation side)
# ---------------------------------------------------------------------------


def _attach_untracked(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing segment without the resource tracker taking it over.

    The writer owns the segment. On Python 3.10-3.12 an attaching process also
    registers it with its resource tracker, which then unlinks the segment when
    that process exits (python/cpython#82300), and an explicit unregister
    confuses a tracker shared with the writer's process. Attaching with the
    registration disabled sidesteps both; 3.13 has ``track=False`` for this.
    """
    try:
        return shared_memory.SharedMemory(name=name, create=False, track=False)  # type: ignore[call-arg]
    except TypeError:
        pass
    register = resource_tracker.register
    resource_tracker.register = lambda *args, **kwargs: None  # type: ignore[assignment]
    try:
        return shared_memory.SharedMemory(name=name, create=False)
    finally:
        resource_tracker.register = register  # type: ignore[assignment]


def _create_segment(name: str, size: int) -> shared_memory.SharedMemory:
    """Create the segment, replacing one left behind by a writer that did not close.

    A simulation killed with SIGKILL leaves its segment in /dev/shm. The next
    writer of the same name takes it over, with a warning; two live writers
    sharing a name is a configuration error this does not detect.
    """
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        _warn_logger.warning("state plane %s: a segment of that name exists; replacing it", name)
        stale = _attach_untracked(name)
        stale.close()
        # unlink() unregisters the name with the resource tracker; register it first so
        # the tracker does not complain about a name it never saw.
        try:
            resource_tracker.register(stale._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - tracker variants
            pass
        stale.unlink()
        return shared_memory.SharedMemory(name=name, create=True, size=size)


class SharedStateWriter:
    """Create the segment and write one :class:`FleetFrame` per call.

    Args:
        name: Segment name; readers attach by it. Also names the pipes.
        capacity: Most rows a frame may hold. A larger frame raises ``ValueError``.
        slots: Ring length. ``every_step`` buffers this many steps before the writer waits.
        mode: ``"every_step"`` or ``"latest"``.
        wait_timeout: Seconds an ``every_step`` writer waits for a slow reader before overwriting.
        timestep: Recorded in the header for readers that derive rates from it.
    """

    def __init__(
        self,
        name: str,
        capacity: int,
        *,
        slots: int = DEFAULT_SLOTS,
        mode: str = MODE_EVERY_STEP,
        wait_timeout: float = DEFAULT_WAIT_TIMEOUT_S,
        timestep: float = 0.0,
        name_width: int = DEFAULT_NAME_WIDTH,
        reader_slots: int = DEFAULT_READER_SLOTS,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.name = name
        self.mode = mode
        self.wait_timeout = float(wait_timeout)
        self.layout = StateLayout(slots=slots, capacity=capacity, name_width=name_width, reader_slots=reader_slots)
        self._shm = _create_segment(name, self.layout.total_bytes)
        self._buf = self._shm.buf
        self._header = np.ndarray((), dtype=_HEADER_DTYPE, buffer=self._buf, offset=0)
        self._readers = np.ndarray(
            (reader_slots,), dtype=_READER_DTYPE, buffer=self._buf, offset=self.layout.reader_table_offset
        )
        self._names = np.ndarray((capacity,), dtype=f"S{name_width}", buffer=self._buf, offset=self.layout.name_table_offset)
        self._slot_meta = [
            np.ndarray((), dtype=_SLOT_META_DTYPE, buffer=self._buf, offset=self.layout.slot_offset(i)) for i in range(slots)
        ]
        self._slot_rows = [
            np.ndarray(
                (),
                dtype=self.layout.rows_dtype,
                buffer=self._buf,
                offset=self.layout.slot_offset(i) + _SLOT_META_DTYPE.itemsize,
            )
            for i in range(slots)
        ]
        self._readers[:] = 0
        self._readers["tail"] = -1
        self._names[:] = b""
        for meta in self._slot_meta:
            meta["seq_begin"] = -1
            meta["seq_end"] = -1
        self._header["magic"] = MAGIC
        self._header["version"] = LAYOUT_VERSION
        self._header["slots"] = slots
        self._header["capacity"] = capacity
        self._header["name_width"] = name_width
        self._header["reader_slots"] = reader_slots
        self._header["timestep"] = float(timestep)
        self._header["head"] = -1
        self._header["generation"] = 0
        self._header["mode"] = 0 if mode == MODE_EVERY_STEP else 1
        self._header["wait_timeout"] = self.wait_timeout
        _owned_names.add(name)
        self._seq = -1
        self._generation = 0
        self._last_ids: Optional[np.ndarray] = None
        self._last_names: Optional[tuple[str, ...]] = None
        self._overrun_warned = False
        self.overruns = 0
        _make_fifo(_pipe_path(name, "w"))
        self._ack_fd = _open_fifo_for_reading(_pipe_path(name, "w"))
        self._reader_fds: dict[int, int] = {}
        self._closed = False

    # -- properties ---------------------------------------------------------

    @property
    def seq(self) -> int:
        """Sequence number of the last committed snapshot (-1 before the first)."""
        return self._seq

    @property
    def generation(self) -> int:
        return self._generation

    def active_readers(self) -> list[int]:
        """Reader slots that are registered and have a fresh heartbeat."""
        now = time.time()
        stale_after = 2.0 * self.wait_timeout
        active = []
        for i in range(self.layout.reader_slots):
            if not self._readers[i]["active"]:
                continue
            if now - float(self._readers[i]["heartbeat"]) > stale_after:
                _warn_logger.warning("state plane %s: reader %d stopped heartbeating; dropping it", self.name, i)
                self._readers[i]["active"] = 0
                self._readers[i]["tail"] = -1
                fd = self._reader_fds.pop(i, None)
                if fd is not None:
                    os.close(fd)
                continue
            active.append(i)
        return active

    # -- writing ------------------------------------------------------------

    def write(self, frame: FleetFrame) -> int:
        """Store *frame* in the next slot and return its sequence number.

        In ``every_step`` mode this waits while the ring is full and a reader is
        registered, up to ``wait_timeout``; see the module docstring.
        """
        if self._closed:
            raise RuntimeError("state plane writer is closed")
        n = frame.n_rows
        if n > self.layout.capacity:
            raise ValueError(f"frame has {n} rows but the state plane holds at most {self.layout.capacity}")
        readers = self.active_readers()
        if self.mode == MODE_EVERY_STEP and readers:
            self._wait_for_room(readers)
        self._bump_generation_if_changed(frame)
        seq = self._seq + 1
        index = seq % self.layout.slots
        meta = self._slot_meta[index]
        rows = self._slot_rows[index]
        meta["seq_begin"] = seq
        meta["seq_end"] = -1
        rows["object_id"][:n] = frame.object_id
        rows["pos"][:n] = frame.pos
        rows["orn"][:n] = frame.orn
        rows["lin_vel"][:n] = frame.lin_vel
        rows["ang_vel"][:n] = frame.ang_vel
        flags = np.zeros(n, dtype=np.uint8)
        flags |= frame.is_moving.astype(np.uint8) * FLAG_MOVING
        flags |= frame.is_charging.astype(np.uint8) * FLAG_CHARGING
        flags |= (~np.isnan(frame.battery_soc)).astype(np.uint8) * FLAG_HAS_BATTERY
        rows["flags"][:n] = flags
        rows["battery"][:n] = frame.battery_soc
        meta["sim_time"] = float(frame.sim_time)
        meta["step"] = int(frame.step)
        meta["generation"] = self._generation
        meta["n_rows"] = n
        meta["seq_end"] = seq
        self._header["head"] = seq
        self._seq = seq
        for i in readers:
            _poke(self._reader_fd(i))
        return seq

    def _bump_generation_if_changed(self, frame: FleetFrame) -> None:
        ids = frame.object_id
        if (
            self._last_ids is not None
            and self._last_ids.shape == ids.shape
            and np.array_equal(self._last_ids, ids)
            and self._last_names == frame.names
        ):
            return
        n = frame.n_rows
        encoded = [name.encode("utf-8")[: self.layout.name_width] for name in frame.names]
        self._names[:n] = encoded
        self._names[n:] = b""
        self._generation += 1
        self._header["generation"] = self._generation
        self._last_ids = ids.copy()
        self._last_names = tuple(frame.names)

    def _wait_for_room(self, readers: list[int]) -> None:
        slots = self.layout.slots
        deadline = time.monotonic() + self.wait_timeout
        while True:
            min_tail = min(int(self._readers[i]["tail"]) for i in readers)
            if self._seq + 1 - min_tail <= slots:
                if self._overrun_warned:
                    _warn_logger.info("state plane %s: readers caught up; every_step delivery resumed", self.name)
                    self._overrun_warned = False
                return
            if self._overrun_warned:
                # Already overwriting: keep going at full speed until the readers catch up.
                self.overruns += 1
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.overruns += 1
                _warn_logger.warning(
                    "state plane %s: a reader did not advance within %.1f s; overwriting the oldest "
                    "snapshot until it catches up",
                    self.name,
                    self.wait_timeout,
                )
                self._overrun_warned = True
                return
            _wait(self._ack_fd, remaining)

    def _reader_fd(self, index: int) -> Optional[int]:
        fd = self._reader_fds.get(index)
        if fd is None:
            fd = _open_fifo_for_writing(_pipe_path(self.name, f"r{index}"))
            if fd is not None:
                self._reader_fds[index] = fd
        return fd

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Remove the segment and the pipes. Readers still attached see stale data and then fail."""
        if self._closed:
            return
        self._closed = True
        for fd in self._reader_fds.values():
            os.close(fd)
        self._reader_fds.clear()
        os.close(self._ack_fd)
        try:
            os.unlink(_pipe_path(self.name, "w"))
        except FileNotFoundError:
            pass
        del self._slot_rows, self._slot_meta, self._names, self._readers, self._header, self._buf
        self._shm.close()
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass
        _owned_names.discard(self.name)

    def __enter__(self) -> "SharedStateWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reader (any process)
# ---------------------------------------------------------------------------


class SharedStateReader:
    """Attach to a writer's segment by name and read snapshots.

    ``read_next`` follows the stream in order (``every_step``); ``read_latest``
    takes the newest committed slot. Both return ``None`` on timeout. Call
    :meth:`close` so the writer stops waiting for this reader.
    """

    def __init__(self, name: str, *, heartbeat: bool = True) -> None:
        self.name = name
        self._shm = _attach_untracked(name)
        self._buf = self._shm.buf
        self._header = np.ndarray((), dtype=_HEADER_DTYPE, buffer=self._buf, offset=0)
        if int(self._header["magic"]) != MAGIC:
            raise ValueError(f"{name!r} is not a PyBulletFleet state plane")
        if int(self._header["version"]) != LAYOUT_VERSION:
            raise ValueError(
                f"state plane {name!r} has layout version {int(self._header['version'])}, this reader knows {LAYOUT_VERSION}"
            )
        self.layout = StateLayout(
            slots=int(self._header["slots"]),
            capacity=int(self._header["capacity"]),
            name_width=int(self._header["name_width"]),
            reader_slots=int(self._header["reader_slots"]),
        )
        self.mode = MODE_EVERY_STEP if int(self._header["mode"]) == 0 else MODE_LATEST
        self.timestep = float(self._header["timestep"])
        # The writer drops a reader whose heartbeat is older than twice its wait_timeout,
        # so a waiting reader refreshes it well within that.
        self._heartbeat_interval = max(0.05, float(self._header["wait_timeout"]) / 2.0)
        self._readers = np.ndarray(
            (self.layout.reader_slots,), dtype=_READER_DTYPE, buffer=self._buf, offset=self.layout.reader_table_offset
        )
        self._names = np.ndarray(
            (self.layout.capacity,), dtype=f"S{self.layout.name_width}", buffer=self._buf, offset=self.layout.name_table_offset
        )
        self._slot_meta = [
            np.ndarray((), dtype=_SLOT_META_DTYPE, buffer=self._buf, offset=self.layout.slot_offset(i))
            for i in range(self.layout.slots)
        ]
        self._slot_rows = [
            np.ndarray(
                (),
                dtype=self.layout.rows_dtype,
                buffer=self._buf,
                offset=self.layout.slot_offset(i) + _SLOT_META_DTYPE.itemsize,
            )
            for i in range(self.layout.slots)
        ]
        self.index = self._register()
        self._heartbeat = heartbeat
        self._names_cache: tuple[str, ...] = ()
        self._names_generation = -1
        _make_fifo(_pipe_path(name, f"r{self.index}"))
        self._wake_fd = _open_fifo_for_reading(_pipe_path(name, f"r{self.index}"))
        self._ack_fd: Optional[int] = None
        self._closed = False

    def _register(self) -> int:
        for i in range(self.layout.reader_slots):
            if not self._readers[i]["active"]:
                # Start with the newest snapshot that already exists, then follow the stream.
                self._readers[i]["tail"] = max(-1, int(self._header["head"]) - 1)
                self._readers[i]["heartbeat"] = time.time()
                self._readers[i]["active"] = 1
                return i
        raise RuntimeError(f"state plane {self.name!r} has no free reader slot ({self.layout.reader_slots} in use)")

    # -- properties ---------------------------------------------------------

    @property
    def head(self) -> int:
        """Sequence number of the newest committed snapshot."""
        return int(self._header["head"])

    @property
    def tail(self) -> int:
        """Sequence number of the last snapshot this reader took."""
        return int(self._readers[self.index]["tail"])

    def is_stale(self) -> bool:
        """Whether the writer has replaced the segment this reader attached to.

        A writer that starts while an older segment of the same name is still
        in place unlinks it and creates a new one; a reader attached to the old
        one keeps a mapping nobody writes to. Compare the attached segment with
        the one the name now resolves to.
        """
        try:
            current = _attach_untracked(self.name)
        except FileNotFoundError:
            return True
        try:
            return os.fstat(current._fd).st_ino != os.fstat(self._shm._fd).st_ino  # type: ignore[attr-defined]
        except (AttributeError, OSError):  # pragma: no cover - platform without fds
            return False
        finally:
            current.close()

    # -- reading ------------------------------------------------------------

    def read_next(self, timeout: Optional[float] = None) -> Optional[Snapshot]:
        """The snapshot after the last one taken, waiting up to *timeout* seconds for it.

        Returns ``None`` on timeout. If the writer overwrote the slot before
        this reader got to it (an overrun), the oldest snapshot still in the
        ring is returned and the gap is logged.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._touch()
            head = self.head
            wanted = self.tail + 1
            if head >= wanted:
                oldest = head - self.layout.slots + 1
                if wanted < oldest:
                    _warn_logger.warning(
                        "state plane %s: reader %d missed snapshots %d-%d", self.name, self.index, wanted, oldest - 1
                    )
                    wanted = oldest
                snapshot = self._read_slot(wanted)
                if snapshot is not None:
                    self._advance(wanted)
                    return snapshot
                # Torn or already overwritten: let the writer finish, then re-evaluate.
                if not self._pause_briefly(deadline):
                    return None
                continue
            if not self._wait_for_wake(deadline):
                return None

    def read_latest(self, timeout: Optional[float] = None) -> Optional[Snapshot]:
        """The newest committed snapshot, waiting up to *timeout* seconds for one newer than the last taken."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._touch()
            head = self.head
            if head > self.tail:
                snapshot = self._read_slot(head)
                if snapshot is not None:
                    self._advance(head)
                    return snapshot
                if not self._pause_briefly(deadline):
                    return None
                continue
            if not self._wait_for_wake(deadline):
                return None

    @staticmethod
    def _pause_briefly(deadline: Optional[float]) -> bool:
        """A short sleep after a torn read; False once *deadline* passes."""
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(0.0002)
        return True

    def _wait_for_wake(self, deadline: Optional[float]) -> bool:
        """Wait for the writer's poke in heartbeat-sized slices; False once *deadline* passes."""
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return False
        slice_s = self._heartbeat_interval if remaining is None else min(remaining, self._heartbeat_interval)
        _wait(self._wake_fd, slice_s)
        return True

    def _read_slot(self, seq: int) -> Optional[Snapshot]:
        if seq < 0:
            return None
        index = seq % self.layout.slots
        meta = self._slot_meta[index]
        rows = self._slot_rows[index]
        if int(meta["seq_end"]) != seq:
            return None
        n = int(meta["n_rows"])
        generation = int(meta["generation"])
        snapshot = Snapshot(
            seq=seq,
            generation=generation,
            sim_time=float(meta["sim_time"]),
            step=int(meta["step"]),
            names=self._names_for(generation, n),
            object_id=rows["object_id"][:n].copy(),
            pos=rows["pos"][:n].copy(),
            orn=rows["orn"][:n].copy(),
            lin_vel=rows["lin_vel"][:n].copy(),
            ang_vel=rows["ang_vel"][:n].copy(),
            is_moving=(rows["flags"][:n] & FLAG_MOVING).astype(bool),
            battery_soc=rows["battery"][:n].copy(),
            is_charging=(rows["flags"][:n] & FLAG_CHARGING).astype(bool),
        )
        if int(meta["seq_begin"]) != seq or int(meta["seq_end"]) != seq:
            return None  # written over while we copied
        return snapshot

    def _names_for(self, generation: int, n: int) -> tuple[str, ...]:
        if generation == self._names_generation and len(self._names_cache) >= n:
            return self._names_cache[:n]
        while True:
            before = int(self._header["generation"])
            names = tuple(bytes(raw).rstrip(b"\x00").decode("utf-8", "replace") for raw in self._names[:n])
            if int(self._header["generation"]) == before:
                self._names_cache = names
                self._names_generation = before
                return names

    def _advance(self, seq: int) -> None:
        self._readers[self.index]["tail"] = seq
        if self._ack_fd is None:
            self._ack_fd = _open_fifo_for_writing(_pipe_path(self.name, "w"))
        _poke(self._ack_fd)

    def _touch(self) -> None:
        if not self._heartbeat:
            return
        if not self._readers[self.index]["active"]:
            # The writer dropped this reader as stale (a long pause); take the slot back.
            self._readers[self.index]["tail"] = max(self.tail, int(self._header["head"]) - 1)
            self._readers[self.index]["active"] = 1
            _warn_logger.warning("state plane %s: reader %d re-registered after being dropped", self.name, self.index)
        self._readers[self.index]["heartbeat"] = time.time()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Leave the reader table and detach; the writer no longer waits for this reader."""
        if self._closed:
            return
        self._closed = True
        try:
            self._readers[self.index]["active"] = 0
            self._readers[self.index]["tail"] = -1
        except Exception:  # pragma: no cover - the writer may be gone already
            pass
        if self._ack_fd is not None:
            _poke(self._ack_fd)
            os.close(self._ack_fd)
        os.close(self._wake_fd)
        try:
            os.unlink(_pipe_path(self.name, f"r{self.index}"))
        except FileNotFoundError:
            pass
        del self._slot_rows, self._slot_meta, self._names, self._readers, self._header, self._buf
        self._shm.close()

    def __enter__(self) -> "SharedStateReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Simulation hook
# ---------------------------------------------------------------------------


class StatePlanePublisher:
    """Write the fleet's state to a shared-memory plane after every simulation step.

    Args:
        sim_core: The simulation whose agents are published.
        name: Segment name readers attach by.
        capacity: Row capacity; defaults to twice the agent count at install, at least 64.
        slots, mode, wait_timeout: As for :class:`SharedStateWriter`.
        publish_rate: Snapshots per simulated second (``None`` publishes every step).

    Example::

        publisher = StatePlanePublisher(sim, name="pbf_state", mode="latest")
        publisher.install()
        ...
        publisher.close()
    """

    def __init__(
        self,
        sim_core: Any,
        name: str = "pbf_state",
        *,
        capacity: Optional[int] = None,
        slots: int = DEFAULT_SLOTS,
        mode: str = MODE_EVERY_STEP,
        wait_timeout: float = DEFAULT_WAIT_TIMEOUT_S,
        publish_rate: Optional[float] = None,
    ) -> None:
        self.sim_core = sim_core
        if capacity is None:
            capacity = max(64, 2 * len(getattr(sim_core, "agents", ())))
        timestep = float(getattr(getattr(sim_core, "params", None), "timestep", 0.0))
        self.writer = SharedStateWriter(name, capacity, slots=slots, mode=mode, wait_timeout=wait_timeout, timestep=timestep)
        self._min_interval = None if not publish_rate else 1.0 / float(publish_rate)
        self._last_publish_time = -math.inf
        self._installed = False
        self.frames_written = 0
        self.write_seconds = 0.0

    def install(self) -> "StatePlanePublisher":
        """Start publishing on the simulation's POST_STEP."""
        from pybullet_fleet.events import SimEvents

        if not self._installed:
            self.sim_core.events.on(SimEvents.POST_STEP, self._on_post_step)
            self._installed = True
        return self

    def _on_post_step(self, sim_time: float = 0.0, **_: Any) -> None:
        if self._min_interval is not None and sim_time - self._last_publish_time < self._min_interval - 1e-9:
            return
        self._last_publish_time = sim_time
        self.publish()

    def publish(self) -> int:
        """Write the current state now (the step hook calls this)."""
        started = time.perf_counter()
        frame = collect_fleet_frame(self.sim_core)
        seq = self.writer.write(frame)
        self.write_seconds += time.perf_counter() - started
        self.frames_written += 1
        return seq

    def close(self) -> None:
        """Stop publishing and remove the segment."""
        from pybullet_fleet.events import SimEvents

        if self._installed:
            self.sim_core.events.off(SimEvents.POST_STEP, self._on_post_step)
            self._installed = False
        self.writer.close()


def snapshot_names(snapshot: Snapshot, ids: Iterable[int]) -> dict[int, str]:
    """Map object ids to names for one snapshot (a convenience for readers)."""
    by_id = dict(zip(snapshot.object_id.tolist(), snapshot.names))
    return {int(i): by_id.get(int(i), "") for i in ids}
