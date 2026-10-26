"""SoA kernel prototype: same workload as steady_bench (N robots on 2m grid, all moving),
diff-drive-ish forward motion with accel limit, AABB from pose, uniform-grid broad phase,
circle narrow phase with margin. Pure numpy, one call per phase per step."""

import statistics
import sys
import time
import numpy as np

n = int(sys.argv[1])
dt = 0.1
side = int(np.ceil(np.sqrt(n)))
ij = np.stack(np.unravel_index(np.arange(n), (side, side)), 1).astype(np.float64)
pos = np.zeros((n, 3))
pos[:, :2] = ij * 2.0
pos[:, 2] = 0.1
yaw = np.zeros(n)
v = np.zeros(n)
goal = pos[:, :2] + np.array([200.0, 0.0])
vmax, amax, radius, margin, cell = 1.5, 2.0, 0.5 * np.sqrt(2) * 0.5, 0.02, 2.0
moving = np.ones(n, bool)


def advance():
    global v
    d = goal - pos[:, :2]
    dist = np.hypot(d[:, 0], d[:, 1])
    vstop = np.sqrt(2 * amax * dist)
    v = np.minimum(np.minimum(v + amax * dt, vmax), vstop)
    step = np.minimum(v * dt, dist)
    u = d / np.maximum(dist, 1e-9)[:, None]
    pos[:, :2] += u * step[:, None]
    yaw[:] = np.arctan2(u[:, 1], u[:, 0])
    arrived = dist - step < 1e-6
    moving[:] = ~arrived
    return np.flatnonzero(arrived)  # event rows for Python


def quats():
    h = 0.5 * yaw
    return np.stack([np.zeros(n), np.zeros(n), np.sin(h), np.cos(h)], 1)


def broad_narrow():
    r = radius + margin
    c = np.floor(pos[:, :2] / cell).astype(np.int64)
    key = c[:, 0] * 1_000_003 + c[:, 1]
    order = np.argsort(key, kind="stable")
    sk = key[order]
    out_i, out_j = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nk = (c[:, 0] + dx) * 1_000_003 + (c[:, 1] + dy)
            lo = np.searchsorted(sk, nk, "left")
            hi = np.searchsorted(sk, nk, "right")
            cnt = hi - lo
            if cnt.sum() == 0:
                continue
            ii = np.repeat(np.arange(n), cnt)
            tot = int(cnt.sum())
            starts = np.cumsum(cnt) - cnt
            jj = order[np.repeat(lo - starts, cnt) + np.arange(tot)]
            m = ii < jj
            out_i.append(ii[m])
            out_j.append(jj[m])
    i = np.concatenate(out_i)
    j = np.concatenate(out_j)
    d = pos[i, :2] - pos[j, :2]
    hit = (d * d).sum(1) <= (2 * r) ** 2
    return i[hit], j[hit], len(i)


for _ in range(20):
    advance()
    quats()
    broad_narrow()
ta, tq, tb = [], [], []
for _ in range(150):
    t0 = time.perf_counter()
    advance()
    t1 = time.perf_counter()
    quats()
    t2 = time.perf_counter()
    hi, hj, ncand = broad_narrow()
    t3 = time.perf_counter()
    ta.append((t1 - t0) * 1e3)
    tq.append((t2 - t1) * 1e3)
    tb.append((t3 - t2) * 1e3)
tot = statistics.mean(ta) + statistics.mean(tq) + statistics.mean(tb)
print(
    f"N={n} advance={statistics.mean(ta):.3f}ms quat={statistics.mean(tq):.3f}ms broad+narrow={statistics.mean(tb):.3f}ms "
    f"cand={ncand} hits={len(hi)} total={tot:.3f}ms RTF={100/tot:.0f}x"
)
