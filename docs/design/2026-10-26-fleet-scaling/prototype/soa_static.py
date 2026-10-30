"""SoA prototype of static_bench.py: N movers among M static boxes, pure numpy.

Statics are keyed into the grid once (sorted cell keys). Each step re-keys only the movers,
finds mover-static candidates by searchsorted into the static keys (3x3 neighbour cells)
and mover-mover candidates among the movers, then runs a z-interval + margin-enlarged AABB
narrow phase. Cells are 2 m and every box here is at most 1 m, so one key per body suffices.

Usage: soa_static.py <n_movers> <field_width_m>
"""

import statistics
import sys
import time
import numpy as np

n = int(sys.argv[1])
width = int(sys.argv[2])
dt, cell, margin = 0.1, 2.0, 0.02
vmax, amax = 1.5, 2.0
ROBOT_HE = np.array([0.25, 0.25, 0.25])

# --- static field: tiles on a 1 m grid, bins on 3 of 4 tiles of the odd rows ---
ii, jj = np.meshgrid(np.arange(-2, 231), np.arange(-2, width + 1), indexing="ij")
ii, jj = ii.ravel().astype(float), jj.ravel().astype(float)
tiles = np.stack([ii, jj, np.full_like(ii, -0.3)], 1)
bmask = (jj.astype(int) % 2 == 1) & (ii.astype(int) % 4 < 3)
bins = np.stack([ii[bmask], jj[bmask], np.full(bmask.sum(), 0.17)], 1)
s_pos = np.concatenate([tiles, bins])
s_he = np.concatenate([np.tile([0.5, 0.5, 0.05], (len(tiles), 1)), np.tile([0.22, 0.33, 0.17], (len(bins), 1))])
s_lo, s_hi = s_pos - s_he - margin / 2, s_pos + s_he + margin / 2
m = len(s_pos)

# --- movers on a 2 m grid, driving +x for 200 m ---
side = int(np.ceil(np.sqrt(n)))
ij = np.stack(np.unravel_index(np.arange(n), (side, side)), 1).astype(np.float64)
pos = np.zeros((n, 3))
pos[:, :2] = ij * 2.0
pos[:, 2] = 0.1
v = np.zeros(n)
goal = pos[:, :2] + np.array([200.0, 0.0])

NX = 1 << 20
ORG = np.array([-4.0, -4.0])


def key_of(p):
    c = np.floor((p[:, :2] - ORG) / cell).astype(np.int64)
    return c[:, 0] * NX + c[:, 1]


s_key = key_of(s_pos)
s_order = np.argsort(s_key, kind="stable")
s_key_sorted = s_key[s_order]
OFFS = np.array([dx * NX + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)], dtype=np.int64)


def advance():
    global v
    d = goal - pos[:, :2]
    dist = np.hypot(d[:, 0], d[:, 1])
    v = np.minimum(np.minimum(v + amax * dt, vmax), np.sqrt(2 * amax * dist))
    step = np.minimum(v * dt, dist)
    nz = dist > 1e-9
    pos[nz, :2] += d[nz] / dist[nz, None] * step[nz, None]


def expand(lo, hi):
    """Row ids repeated per candidate, and the matching positions in the sorted array."""
    cnt = hi - lo
    rows = np.repeat(np.arange(len(lo)), cnt)
    idx = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    return rows, idx


def broad_narrow():
    mk = key_of(pos)
    lo_r, hi_r = pos - ROBOT_HE - margin / 2, pos + ROBOT_HE + margin / 2
    # mover-static
    q = (mk[:, None] + OFFS[None, :]).ravel()
    lo = np.searchsorted(s_key_sorted, q, "left")
    hi = np.searchsorted(s_key_sorted, q, "right")
    rows, idx = expand(lo, hi)
    rows //= 1
    ri = rows // len(OFFS)
    sj = s_order[idx]
    n_ms = len(ri)
    ov = np.all((lo_r[ri] <= s_hi[sj]) & (hi_r[ri] >= s_lo[sj]), axis=1)
    hits_ms = ov.sum()
    # mover-mover
    order = np.argsort(mk, kind="stable")
    mks = mk[order]
    lo = np.searchsorted(mks, q, "left")
    hi = np.searchsorted(mks, q, "right")
    rows, idx = expand(lo, hi)
    ri = rows // len(OFFS)
    rj = order[idx]
    keep = ri < rj
    ri, rj = ri[keep], rj[keep]
    ov = np.all((lo_r[ri] <= hi_r[rj]) & (hi_r[ri] >= lo_r[rj]), axis=1)
    return n_ms, int(hits_ms), len(ri), int(ov.sum())


for _ in range(20):
    advance()
    broad_narrow()
ta, tb = [], []
for _ in range(150):
    t0 = time.perf_counter()
    advance()
    t1 = time.perf_counter()
    stats = broad_narrow()
    t2 = time.perf_counter()
    ta.append((t1 - t0) * 1e3)
    tb.append((t2 - t1) * 1e3)
ma, mb = statistics.median(ta), statistics.median(tb)
print(
    f"movers={n} statics={m} advance={ma:.3f}ms broad+narrow={mb:.3f}ms "
    f"cand(mover-static)={stats[0]} hits={stats[1]} cand(mover-mover)={stats[2]} hits={stats[3]} "
    f"total={ma + mb:.3f}ms RTF={100 / (ma + mb):.0f}x (median of steps)"
)
