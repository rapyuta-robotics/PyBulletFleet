# Shared-Memory State Plane

The state plane lets another process read every robot's state each step
without running inside the simulation. The simulation writes a snapshot of
its agents into a ring of slots in a `multiprocessing.shared_memory` segment
after each step; readers in other processes copy the slot they need. The
ROS 2 bridge, a viewer or a recorder can run as such a reader, so none of
their work lands on the stepping thread.

Writing a 1000-robot snapshot costs about 20 µs, a 10,000-robot one under
0.15 ms; a reader copies them out in about 0.15 ms and 0.9 ms
(`benchmark/state_plane_bench.py`). Collecting the arrays from the agents
(`collect_fleet_frame`) costs about 2 ms per 1000 robots until the core keeps
its state in arrays: it reads each agent's pose, velocity, flags and battery
and converts each column once.

## Publishing from a simulation

```python
from pybullet_fleet import MultiRobotSimulationCore
from pybullet_fleet.state_plane import StatePlanePublisher

sim = MultiRobotSimulationCore.from_yaml("config/config.yaml")
publisher = StatePlanePublisher(sim, name="pbf_state", mode="every_step").install()
try:
    sim.run_simulation()
finally:
    publisher.close()  # removes the segment; do this before the process exits
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `name` | `"pbf_state"` | Segment name; readers attach by it. One writer per name on a host. |
| `capacity` | twice the agent count, at least 64 | Most rows a snapshot may hold. A larger fleet raises `ValueError` on write. |
| `slots` | 8 | Ring length. `every_step` buffers this many steps before the writer waits. |
| `mode` | `"every_step"` | `every_step` delivers every snapshot to every reader; `latest` never waits. |
| `wait_timeout` | 1.0 s | How long an `every_step` writer waits for a slow reader before overwriting. |
| `publish_rate` | every step | Snapshots per simulated second. |

## Reading from another process

```python
from pybullet_fleet.state_plane import SharedStateReader

with SharedStateReader("pbf_state") as reader:
    while True:
        snapshot = reader.read_next(timeout=1.0)   # every_step: the next one in order
        if snapshot is None:
            continue                                # nothing within a second
        print(snapshot.seq, snapshot.sim_time, snapshot.names[0], snapshot.pos[0])
```

`read_latest(timeout)` returns the newest committed snapshot instead and
skips the rest; use it with `mode: latest`, or with `every_step` when the
reader only needs a sample.

A `Snapshot` carries `seq`, `sim_time`, `step`, `generation`, `names` and
one row per agent in `object_id`, `pos` (x, y, z), `orn` (x, y, z, w),
`lin_vel`, `ang_vel`, `is_moving`, `battery_soc` (NaN without a battery) and
`is_charging`. `generation` increases whenever the set of agents changes; a
reader that keeps per-robot state keyed by name re-reads `names` then.

## Delivery guarantees

- **`every_step`:** each registered reader sees every snapshot, in order,
  as long as it keeps up. When the ring is full the writer waits for the
  slowest reader, up to `wait_timeout`; past that it logs one warning and
  overwrites the oldest slots at full speed until the readers catch up, and
  a reader that fell behind gets the oldest surviving snapshot and a warning
  naming the sequence numbers it missed. Without any reader the writer never
  waits.
- **`latest`:** the writer never waits. A reader takes the newest committed
  snapshot.
- A reader that attaches while snapshots already exist starts with the
  newest of them, then follows the stream from there.
- A reader that stops reading for longer than twice `wait_timeout` is
  dropped by the writer (its heartbeat goes stale) so it cannot stall the
  simulation; it re-registers by itself on its next read.
- A snapshot is never returned half-written: each slot is a seqlock, and a
  reader that catches a slot mid-write reads it again.

## How it works, and its limits

The data lives in one shared-memory segment (header, a name table, then the
slots). Wakeups and backpressure go through named pipes under the temp
directory (`$PYBULLET_FLEET_STATE_PLANE_DIR` overrides it): the writer pokes
each reader's pipe after a commit, and a reader pokes the writer's pipe after
it advances. A blocking read on a pipe is an OS wait, not a busy loop, and the
pipe syscalls order the shared-memory stores before the wakeup, which Python's
numpy stores alone do not guarantee on every CPU.

- Linux and macOS. Windows has no named pipes of this kind; use WSL2.
- At most 8 readers per plane (`reader_slots` on the writer).
- The writer owns the segment and the pipes and removes them in `close()`.
  A reader only attaches; on Python 3.10-3.12 it attaches without
  registering with the resource tracker, which would otherwise unlink the
  segment when the reader exits ([python/cpython#82300](https://github.com/python/cpython/issues/82300)).
- `names` are stored as UTF-8 in 64 bytes each; longer names are cut.
