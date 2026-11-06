"""Cost of writing a fleet snapshot to the shared-memory state plane.

Measures the write alone (no reader), then with one reader in another process
taking every snapshot (``every_step``), for 1000 and 10000 rows::

    python3 benchmark/state_plane_bench.py
    python3 benchmark/state_plane_bench.py --rows 1000 5000 --frames 300
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import statistics
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pybullet_fleet.state_plane import FleetFrame, SharedStateReader, SharedStateWriter


def make_frame(n: int, t: float = 0.0, step: int = 0) -> FleetFrame:
    return FleetFrame(
        sim_time=t,
        step=step,
        names=tuple(f"robot{i}" for i in range(n)),
        object_id=np.arange(n),
        pos=np.full((n, 3), t),
        orn=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        lin_vel=np.zeros((n, 3)),
        ang_vel=np.zeros((n, 3)),
        is_moving=np.ones(n, dtype=bool),
        battery_soc=np.full(n, np.nan, dtype=np.float32),
        is_charging=np.zeros(n, dtype=bool),
    )


def _reader(name: str, frames: int, queue) -> None:
    reader = SharedStateReader(name)
    read_ms = []
    received = 0
    try:
        while received < frames:
            started = time.perf_counter()
            snapshot = reader.read_next(timeout=10.0)
            if snapshot is None:
                break
            read_ms.append((time.perf_counter() - started) * 1e3)
            received += 1
    finally:
        reader.close()
    queue.put((received, statistics.median(read_ms) if read_ms else float("nan")))


def bench(rows: int, frames: int) -> dict:
    name = "pbf_bench_" + uuid.uuid4().hex[:8]
    writer = SharedStateWriter(name, rows, slots=8, mode="every_step", wait_timeout=2.0)
    frame = make_frame(rows)
    try:
        writer.write(frame)
        times = []
        for _ in range(frames):
            started = time.perf_counter()
            writer.write(frame)
            times.append((time.perf_counter() - started) * 1e3)
        alone_ms = statistics.median(times)

        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=_reader, args=(name, frames, queue))
        process.start()
        while not writer.active_readers():
            time.sleep(0.01)
        times = []
        for i in range(frames):
            started = time.perf_counter()
            writer.write(make_frame(rows, t=i * 0.1, step=i))
            times.append((time.perf_counter() - started) * 1e3)
        received, read_ms = queue.get(timeout=30.0)
        process.join(timeout=10.0)
        return {
            "rows": rows,
            "frames": frames,
            "write_alone_ms": alone_ms,
            "write_with_reader_ms": statistics.median(times),
            "write_with_reader_p95_ms": sorted(times)[int(0.95 * (len(times) - 1))],
            "reader_received": received,
            "reader_read_ms": read_ms,
            "overruns": writer.overruns,
            "segment_mb": writer.layout.total_bytes / 1e6,
        }
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    for rows in args.rows:
        r = bench(rows, args.frames)
        print(
            f"rows={r['rows']}: write alone {r['write_alone_ms']:.3f} ms, with one reader "
            f"{r['write_with_reader_ms']:.3f} ms (p95 {r['write_with_reader_p95_ms']:.3f}), "
            f"reader copy {r['reader_read_ms']:.3f} ms, received {r['reader_received']}/{r['frames']}, "
            f"overruns {r['overruns']}, segment {r['segment_mb']:.1f} MB"
        )


if __name__ == "__main__":
    main()
