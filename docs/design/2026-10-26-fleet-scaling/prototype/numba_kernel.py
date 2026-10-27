"""Numba variant of soa_kernel: same workload, counting-sort grid, parallel neighbor scan.

The grid is dense (nx * ny cells over the bounding box of the scene), which is fine for
this benchmark's compact layout but not for a sparse fleet on a large map. A production
version would hash cell keys into a bounded table, as soa_kernel.py's sort-based grid does.
"""

import statistics
import sys
import time
import numpy as np
import numba as nb


@nb.njit(parallel=True, fastmath=True, cache=True)
def advance(pos, yaw, v, goal, dt, vmax, amax):
    n = pos.shape[0]
    for k in nb.prange(n):
        dx = goal[k, 0] - pos[k, 0]
        dy = goal[k, 1] - pos[k, 1]
        d = np.sqrt(dx * dx + dy * dy)
        vk = min(min(v[k] + amax * dt, vmax), np.sqrt(2 * amax * d))
        v[k] = vk
        s = min(vk * dt, d)
        if d > 1e-9:
            pos[k, 0] += dx / d * s
            pos[k, 1] += dy / d * s
            yaw[k] = np.arctan2(dy, dx)


@nb.njit(cache=True)
def build_grid(pos, cell, nx, ny, ox, oy):
    n = pos.shape[0]
    cid = np.empty(n, np.int64)
    counts = np.zeros(nx * ny + 1, np.int64)
    for k in range(n):
        cx = min(max(int((pos[k, 0] - ox) / cell), 0), nx - 1)
        cy = min(max(int((pos[k, 1] - oy) / cell), 0), ny - 1)
        c = cx * ny + cy
        cid[k] = c
        counts[c + 1] += 1
    for c in range(nx * ny):
        counts[c + 1] += counts[c]
    fill = counts[:-1].copy()
    items = np.empty(n, np.int64)
    for k in range(n):
        items[fill[cid[k]]] = k
        fill[cid[k]] += 1
    return cid, counts, items


@nb.njit(parallel=True, fastmath=True, cache=True)
def collide(pos, cid, counts, items, nx, ny, r2, hits):
    n = pos.shape[0]
    ncand = np.zeros(n, np.int64)
    for k in nb.prange(n):
        cx = cid[k] // ny
        cy = cid[k] % ny
        h = 0
        m = 0
        for ax in range(max(cx - 1, 0), min(cx + 2, nx)):
            for ay in range(max(cy - 1, 0), min(cy + 2, ny)):
                c = ax * ny + ay
                for t in range(counts[c], counts[c + 1]):
                    j = items[t]
                    if j > k:
                        m += 1
                        dx = pos[k, 0] - pos[j, 0]
                        dy = pos[k, 1] - pos[j, 1]
                        if dx * dx + dy * dy <= r2:
                            h += 1
        hits[k] = h
        ncand[k] = m
    return ncand.sum()


n = int(sys.argv[1])
dt = 0.1
side = int(np.ceil(np.sqrt(n)))
ij = np.stack(np.unravel_index(np.arange(n), (side, side)), 1).astype(np.float64)
pos = np.zeros((n, 3))
pos[:, :2] = ij * 2.0
yaw = np.zeros(n)
v = np.zeros(n)
goal = pos[:, :2] + np.array([200.0, 0.0])
hits = np.zeros(n, np.int64)
r = 0.5 * np.sqrt(2) * 0.5 + 0.02
cell = 2.0


def step():
    advance(pos, yaw, v, goal, dt, 1.5, 2.0)
    ox = pos[:, 0].min() - cell
    oy = pos[:, 1].min() - cell
    nx = int((pos[:, 0].max() - ox) / cell) + 2
    ny = int((pos[:, 1].max() - oy) / cell) + 2
    cid, counts, items = build_grid(pos, cell, nx, ny, ox, oy)
    return collide(pos, cid, counts, items, nx, ny, (2 * r) ** 2, hits)


for _ in range(20):
    step()
ts = []
for _ in range(150):
    t0 = time.perf_counter()
    nc = step()
    ts.append((time.perf_counter() - t0) * 1e3)
m = statistics.median(ts)
print(f"N={n} threads={nb.get_num_threads()} total(p50)={m:.3f}ms cand={nc} RTF={100/m:.0f}x")
