"""Columns of every object's collision state, and the array broad phase over them.

The simulation core keeps one row per registered object here: its AABB, the
local box the AABB is computed from, its collision mode and whether it is a
physics body. Rows are internal (they move when an object is removed); the
core and everything outside it refer to objects by ``object_id``.

The spatial hash over the rows is the one of #94: each registered body is in
every cell its AABB, enlarged on each axis by half the collision clearance of
that axis (not in z for a NORMAL_2D body), touches, and a candidate search
looks only in a body's own cells. It is kept as a sorted array of packed cell
keys with the row of each entry alongside, so a search is a ``searchsorted``
and the pairs of all moved bodies come out of a few array operations.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Collision mode codes of the ``mode`` column.
MODE_NORMAL_3D = 0
MODE_NORMAL_2D = 1
MODE_STATIC = 2
MODE_DISABLED = 3

_CELL_OFFSET = 1 << 20  # cells are packed as unsigned offsets of this size per axis
_CELL_SPAN = 1 << 21

Box = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


def pack_cells(cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> np.ndarray:
    """One int64 key per (x, y, z) cell; sorts first by x, then y, then z."""
    return ((cx + _CELL_OFFSET) * _CELL_SPAN + (cy + _CELL_OFFSET)) * _CELL_SPAN + (cz + _CELL_OFFSET)


def rotation_matrices(orn: np.ndarray) -> np.ndarray:
    """``(k, 3, 3)`` rotation matrices of ``(k, 4)`` xyzw quaternions."""
    x, y, z, w = orn[:, 0], orn[:, 1], orn[:, 2], orn[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.empty((len(orn), 3, 3))
    r[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    r[:, 0, 1] = 2.0 * (xy - wz)
    r[:, 0, 2] = 2.0 * (xz + wy)
    r[:, 1, 0] = 2.0 * (xy + wz)
    r[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    r[:, 1, 2] = 2.0 * (yz - wx)
    r[:, 2, 0] = 2.0 * (xz - wy)
    r[:, 2, 1] = 2.0 * (yz + wx)
    r[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return r


def quaternion_products(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise products ``a * b`` of ``(k, 4)`` xyzw quaternions, as ``p.multiplyTransforms`` composes orientations.

    Computed in double precision; PyBullet's own result agrees to about 1e-7.
    """
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = np.empty((len(a), 4))
    out[:, 0] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 1] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 2] = aw * bz + ax * by - ay * bx + az * bw
    out[:, 3] = aw * bw - ax * bx - ay * by - az * bz
    # The same rotation with the sign PyBullet reports: it reads the product
    # back from a rotation matrix (btMatrix3x3::getRotation), which makes w
    # positive when |w| > 1/2 and otherwise the largest of x, y, z positive.
    w_large = np.abs(out[:, 3]) > 0.5
    lead = np.where(w_large, 3, np.argmax(np.abs(out[:, :3]), axis=1))
    out[out[np.arange(len(out)), lead] < 0.0] *= -1.0
    return out


class CollisionStore:
    """Per-object collision columns and the spatial hash over them."""

    def __init__(self, capacity: int = 256) -> None:
        self._n = 0
        self.row_of: Dict[int, int] = {}
        #: Incremented whenever the columns are reallocated: views into them are then stale.
        self.generation = 0
        self._allocate(max(capacity, 1))
        self.cell_size: Optional[float] = None
        self.half_margin = np.zeros(3)
        # Two sorted entry arrays: the STATIC rows' entries, patched only when
        # a static row changes, and the other rows' entries, rebuilt whenever
        # one of them changes (a driving fleet changes most of them anyway).
        self._static_keys = np.empty(0, np.int64)
        self._static_rows = np.empty(0, np.int64)
        self._moving_keys = np.empty(0, np.int64)
        self._moving_rows = np.empty(0, np.int64)
        self._hash_built = False

    def _allocate(self, capacity: int) -> None:
        self.object_id = np.zeros(capacity, np.int64)
        #: The pose the simulation holds for the object (its source of truth
        #: for a kinematic object; a physics body's as of its last read).
        self.pos = np.zeros((capacity, 3))
        self.orn = np.zeros((capacity, 4))
        self.orn[:, 3] = 1.0
        #: Whether the next flush of the row's pose keeps the body's velocity.
        self.preserve_velocity = np.ones(capacity, bool)
        #: PyBullet does not hold the row's pose yet (a buffered write not mirrored).
        self.mirror_dirty = np.zeros(capacity, bool)
        #: PyBullet must hold the row's pose after every step (a physics body,
        #: an articulated body, a body whose links carry attachments).
        self.needs_mirror = np.zeros(capacity, bool)
        #: The pose of a base-attached object in its carrier's frame (identity when not attached).
        self.attach_pos = np.zeros((capacity, 3))
        self.attach_orn = np.zeros((capacity, 4))
        self.attach_orn[:, 3] = 1.0
        self.aabb_min = np.full((capacity, 3), np.nan)
        self.aabb_max = np.full((capacity, 3), np.nan)
        #: The body's box in its own frame, at identity orientation. NaN until known.
        self.local_center = np.full((capacity, 3), np.nan)
        self.local_half = np.full((capacity, 3), np.nan)
        #: The extents do not rotate (a sphere).
        self.invariant = np.zeros(capacity, bool)
        #: The AABB is read from PyBullet (a physics body, an articulated body), not computed from the pose.
        self.pybullet_aabb = np.zeros(capacity, bool)
        self.mode = np.full(capacity, MODE_NORMAL_3D, np.int8)
        self.physics = np.zeros(capacity, bool)
        self.pickable = np.zeros(capacity, bool)
        #: An AABB has been computed for the row.
        self.has_aabb = np.zeros(capacity, bool)
        #: The row's hash entries do not match its AABB / mode.
        self.hash_dirty = np.zeros(capacity, bool)
        self._cell_lo = np.zeros((capacity, 3), np.int64)
        self._cell_hi = np.zeros((capacity, 3), np.int64)

    # --- rows -------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    @property
    def capacity(self) -> int:
        return len(self.object_id)

    def _grow(self) -> None:
        old = {name: getattr(self, name) for name in self._COLUMNS}
        self._allocate(self.capacity * 2)
        n = self._n
        for name, column in old.items():
            getattr(self, name)[:n] = column[:n]
        self.generation += 1

    _COLUMNS = (
        "object_id",
        "pos",
        "orn",
        "preserve_velocity",
        "mirror_dirty",
        "needs_mirror",
        "attach_pos",
        "attach_orn",
        "aabb_min",
        "aabb_max",
        "local_center",
        "local_half",
        "invariant",
        "pybullet_aabb",
        "mode",
        "physics",
        "pickable",
        "has_aabb",
        "hash_dirty",
        "_cell_lo",
        "_cell_hi",
    )

    def add(
        self,
        object_id: int,
        *,
        mode: int,
        physics: bool,
        pickable: bool,
        pybullet_aabb: bool,
        pos: Sequence[float] = (0.0, 0.0, 0.0),
        orn: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    ) -> int:
        """Add a row for ``object_id`` at the pose, with no AABB yet, and return it."""
        if object_id in self.row_of:
            raise ValueError(f"object {object_id} already has a row")
        if self._n == self.capacity:
            self._grow()
        row = self._n
        self._n += 1
        self.row_of[object_id] = row
        self.object_id[row] = object_id
        self.pos[row] = pos
        self.orn[row] = orn
        self.preserve_velocity[row] = True
        self.mirror_dirty[row] = False
        self.needs_mirror[row] = False
        self.attach_pos[row] = 0.0
        self.attach_orn[row] = (0.0, 0.0, 0.0, 1.0)
        self.aabb_min[row] = np.nan
        self.aabb_max[row] = np.nan
        self.local_center[row] = np.nan
        self.local_half[row] = np.nan
        self.invariant[row] = False
        self.pybullet_aabb[row] = pybullet_aabb
        self.mode[row] = mode
        self.physics[row] = physics
        self.pickable[row] = pickable
        self.has_aabb[row] = False
        self.hash_dirty[row] = True
        return row

    def remove(self, object_id: int) -> Optional[int]:
        """Drop the row of ``object_id``; the last row takes its place.

        Returns the object whose row moved into the freed one (its views into
        the columns are stale), or None when the last row was the one dropped.
        """
        row = self.row_of.pop(object_id)
        last = self._n - 1
        if self._hash_built:
            self._static_keys, self._static_rows = self._drop_row(self._static_keys, self._static_rows, row, last)
            self._moving_keys, self._moving_rows = self._drop_row(self._moving_keys, self._moving_rows, row, last)
        moved: Optional[int] = None
        if last != row:
            for name in self._COLUMNS:
                column = getattr(self, name)
                column[row] = column[last]
            moved = int(self.object_id[row])
            self.row_of[moved] = row
        self._n = last
        return moved

    @staticmethod
    def _drop_row(keys: np.ndarray, rows: np.ndarray, row: int, last: int) -> Tuple[np.ndarray, np.ndarray]:
        """The entries without those of ``row``, with the entries of ``last`` renamed to ``row``."""
        keep = rows != row
        if not keep.all():
            keys, rows = keys[keep], rows[keep]
        if last != row:
            rows[rows == last] = row
        return keys, rows

    def clear(self) -> None:
        self._n = 0
        self.row_of.clear()
        self._static_keys = np.empty(0, np.int64)
        self._static_rows = np.empty(0, np.int64)
        self._moving_keys = np.empty(0, np.int64)
        self._moving_rows = np.empty(0, np.int64)
        self._hash_built = False

    def rows(self, object_ids: Sequence[int]) -> np.ndarray:
        """The rows of the given objects, skipping ids without a row."""
        row_of = self.row_of
        return np.fromiter((row_of[i] for i in object_ids if i in row_of), np.int64)

    # --- AABBs ------------------------------------------------------------

    def set_local_box(self, row: int, local_min: Sequence[float], local_max: Sequence[float], invariant: bool) -> None:
        center = self.local_center[row]
        half = self.local_half[row]
        for axis in range(3):
            lo, hi = float(local_min[axis]), float(local_max[axis])
            center[axis] = (lo + hi) * 0.5
            half[axis] = (hi - lo) * 0.5
        self.invariant[row] = invariant

    def local_box_known(self, row: int) -> bool:
        return not math.isnan(self.local_half[row, 0])

    def set_aabb(self, row: int, aabb: Box) -> None:
        self.aabb_min[row] = aabb[0]
        self.aabb_max[row] = aabb[1]
        self.has_aabb[row] = True
        self.hash_dirty[row] = True

    def set_aabbs_from_poses(self, rows: np.ndarray, pos: np.ndarray, orn: np.ndarray) -> None:
        """AABBs of ``rows`` at the poses, from their local boxes, as PyBullet computes them.

        PyBullet's box of a rotated body is its local box transformed by the
        element-wise absolute rotation (``btTransformAabb``); a sphere's
        extents do not rotate.
        """
        r = rotation_matrices(orn)
        center = pos + np.einsum("kij,kj->ki", r, self.local_center[rows])
        extent = np.einsum("kij,kj->ki", np.abs(r), self.local_half[rows])
        invariant = self.invariant[rows]
        if invariant.any():
            extent[invariant] = self.local_half[rows][invariant]
        self.aabb_min[rows] = center - extent
        self.aabb_max[rows] = center + extent
        self.has_aabb[rows] = True
        self.hash_dirty[rows] = True

    def attached_poses(self, child_rows: np.ndarray, parent_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The world poses of base-attached ``child_rows`` carried by ``parent_rows`` (one parent row per child)."""
        parent_orn = self.orn[parent_rows]
        r = rotation_matrices(parent_orn)
        pos = self.pos[parent_rows] + np.einsum("kij,kj->ki", r, self.attach_pos[child_rows])
        return pos, quaternion_products(parent_orn, self.attach_orn[child_rows])

    def set_aabb_from_pose(self, row: int, pos: Sequence[float], orn: Sequence[float]) -> None:
        """:meth:`set_aabbs_from_poses` for one row, in plain arithmetic (a few microseconds, no array overhead)."""
        x, y, z, w = orn
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        r00, r01, r02 = 1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)
        r10, r11, r12 = 2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)
        r20, r21, r22 = 2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)
        cx, cy, cz = self.local_center[row]
        hx, hy, hz = self.local_half[row]
        px = pos[0] + r00 * cx + r01 * cy + r02 * cz
        py = pos[1] + r10 * cx + r11 * cy + r12 * cz
        pz = pos[2] + r20 * cx + r21 * cy + r22 * cz
        if self.invariant[row]:
            ex, ey, ez = hx, hy, hz
        else:
            ex = abs(r00) * hx + abs(r01) * hy + abs(r02) * hz
            ey = abs(r10) * hx + abs(r11) * hy + abs(r12) * hz
            ez = abs(r20) * hx + abs(r21) * hy + abs(r22) * hz
        aabb_min = self.aabb_min[row]
        aabb_max = self.aabb_max[row]
        aabb_min[0], aabb_min[1], aabb_min[2] = px - ex, py - ey, pz - ez
        aabb_max[0], aabb_max[1], aabb_max[2] = px + ex, py + ey, pz + ez
        self.has_aabb[row] = True
        self.hash_dirty[row] = True

    def aabb(self, row: int) -> Box:
        lo = self.aabb_min[row]
        hi = self.aabb_max[row]
        return (float(lo[0]), float(lo[1]), float(lo[2])), (float(hi[0]), float(hi[1]), float(hi[2]))

    def median_extent(self) -> Optional[float]:
        """The median of the rows' largest AABB extents, or None without any AABB."""
        n = self._n
        have = self.has_aabb[:n]
        if not have.any():
            return None
        extents = (self.aabb_max[:n][have] - self.aabb_min[:n][have]).max(axis=1)
        return float(np.sort(extents)[len(extents) // 2])

    def query_overlapping(self, lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
        """Rows with an AABB overlapping the box ``[lo, hi]``."""
        n = self._n
        if n == 0:
            return np.empty(0, np.int64)
        lo_a = np.asarray(lo, dtype=float)
        hi_a = np.asarray(hi, dtype=float)
        inside = (self.aabb_min[:n] <= hi_a).all(axis=1) & (self.aabb_max[:n] >= lo_a).all(axis=1)
        return np.flatnonzero(inside & self.has_aabb[:n])

    # --- spatial hash -----------------------------------------------------

    def set_hash(self, cell_size: float, half_margin: Sequence[float]) -> None:
        """Set the cell size and the per-axis registration margin; the hash is rebuilt on the next sync."""
        self.cell_size = float(cell_size)
        self.half_margin = np.asarray(half_margin, dtype=float)
        self._hash_built = False

    def registered(self, rows: np.ndarray) -> np.ndarray:
        """Whether each row is in the hash: it has an AABB and collision is not DISABLED."""
        return self.has_aabb[rows] & (self.mode[rows] != MODE_DISABLED)

    def cell_ranges(self, rows: np.ndarray, margin: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """The ``(lo, hi)`` cell index ranges of the rows' AABBs enlarged by ``margin`` (default: the hash margin).

        A NORMAL_2D row is not enlarged in z: a pair with one needs real z
        overlap, and the cell holding that overlap is one both register.
        """
        if self.cell_size is None:
            raise ValueError("the spatial hash has no cell size yet")
        m = np.broadcast_to(self.half_margin if margin is None else margin, (len(rows), 3)).copy()
        m[self.mode[rows] == MODE_NORMAL_2D, 2] = 0.0
        lo = np.floor((self.aabb_min[rows] - m) / self.cell_size).astype(np.int64)
        hi = np.floor((self.aabb_max[rows] + m) / self.cell_size).astype(np.int64)
        return lo, hi

    def cells_of(self, row: int, margin: Optional[Sequence[float]] = None) -> List[Tuple[int, int, int]]:
        """The cells the row's AABB enlarged by ``margin`` (default: the hash margin) touches, in x, y, z order."""
        if self.cell_size is None or not self.has_aabb[row]:
            return []
        rows = np.array([row])
        lo, hi = self.cell_ranges(rows, None if margin is None else np.asarray(margin, dtype=float))
        (x0, y0, z0), (x1, y1, z1) = lo[0].tolist(), hi[0].tolist()
        return [(x, y, z) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1) for z in range(z0, z1 + 1)]

    @staticmethod
    def _entries(rows: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One ``(key, row)`` entry per cell of each row's range, all rows at once."""
        spans = hi - lo + 1
        counts = spans.prod(axis=1)
        if (counts == 1).all():
            return pack_cells(lo[:, 0], lo[:, 1], lo[:, 2]), rows
        total = int(counts.sum())
        which = np.repeat(np.arange(len(rows)), counts)
        offset = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        ny = spans[which, 1]
        nz = spans[which, 2]
        yz = ny * nz
        cx = lo[which, 0] + offset // yz
        rest = offset % yz
        cy = lo[which, 1] + rest // nz
        cz = lo[which, 2] + rest % nz
        return pack_cells(cx, cy, cz), rows[which]

    def sync_hash(self) -> None:
        """Bring the hash up to date with the rows' AABBs and modes."""
        n = self._n
        if self.cell_size is None:
            return
        dirty = np.flatnonzero(self.hash_dirty[:n])
        if self._hash_built and len(dirty) == 0:
            return
        static = self.mode[:n] == MODE_STATIC
        if not self._hash_built:
            rows = np.arange(n)
            self._static_keys, self._static_rows = self._sorted_entries(rows[static])
            self._moving_keys, self._moving_rows = self._sorted_entries(rows[~static])
        else:
            is_dirty = np.zeros(n, bool)
            is_dirty[dirty] = True
            # A row that was static and is no longer, or the reverse, has
            # entries on the side it left: drop them there too.
            static_dirty = dirty[static[dirty]]
            moving_dirty = dirty[~static[dirty]]
            if len(static_dirty) or (is_dirty[self._static_rows]).any():
                keep = ~is_dirty[self._static_rows]
                keys, rows = self._static_keys[keep], self._static_rows[keep]
                new_keys, new_rows = self._sorted_entries(static_dirty)
                if len(new_keys):
                    at = np.searchsorted(keys, new_keys)
                    keys = np.insert(keys, at, new_keys)
                    rows = np.insert(rows, at, new_rows)
                self._static_keys, self._static_rows = keys, rows
            if len(moving_dirty) or (is_dirty[self._moving_rows]).any():
                rows = np.arange(n)
                self._moving_keys, self._moving_rows = self._sorted_entries(rows[~static])
        self.hash_dirty[:n] = False
        self._hash_built = True

    def _sorted_entries(self, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The sorted ``(keys, rows)`` entries of the registered rows among ``rows``; their cell ranges are stored."""
        rows = rows[self.registered(rows)]
        if len(rows) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        lo, hi = self.cell_ranges(rows)
        self._cell_lo[rows] = lo
        self._cell_hi[rows] = hi
        keys, entry_rows = self._entries(rows, lo, hi)
        order = np.argsort(keys, kind="stable")
        return keys[order], entry_rows[order]

    def hash_contents(self) -> Dict[Tuple[int, int, int], set]:
        """The hash as ``{cell: {object_id, ...}}``, for inspection."""
        self.sync_hash()
        contents: Dict[Tuple[int, int, int], set] = {}
        keys = np.concatenate([self._static_keys, self._moving_keys])
        entry_rows = np.concatenate([self._static_rows, self._moving_rows])
        cz = keys % _CELL_SPAN - _CELL_OFFSET
        rest = keys // _CELL_SPAN
        cy = rest % _CELL_SPAN - _CELL_OFFSET
        cx = rest // _CELL_SPAN - _CELL_OFFSET
        ids = self.object_id[entry_rows]
        for x, y, z, object_id in zip(cx.tolist(), cy.tolist(), cz.tolist(), ids.tolist()):
            contents.setdefault((x, y, z), set()).add(object_id)
        return contents

    def candidate_pairs(
        self, rows: np.ndarray, clearance: Sequence[float], *, ignore_static: bool, hybrid: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The pairs of the moved ``rows`` with the bodies sharing one of their cells, within the clearance.

        Each pair comes once, as two arrays of object ids. A pair is within
        the clearance when the AABBs, each axis enlarged by that axis's
        clearance, overlap; no z clearance with a NORMAL_2D body, and no
        clearance at all for a pair with a physics body under ``hybrid``.
        With ``ignore_static`` STATIC bodies take part in no pair.
        """
        self.sync_hash()
        empty = np.empty(0, np.int64)
        if len(rows) == 0 or (len(self._static_keys) == 0 and len(self._moving_keys) == 0):
            return empty, empty
        rows = np.unique(rows)
        active = self.registered(rows)
        if ignore_static:
            active &= self.mode[rows] != MODE_STATIC
        rows = rows[active]
        if len(rows) == 0:
            return empty, empty
        qkeys, qrows = self._entries(rows, self._cell_lo[rows], self._cell_hi[rows])
        found_i = []
        found_j = []
        sides = [(self._moving_keys, self._moving_rows)]
        if not ignore_static:
            sides.append((self._static_keys, self._static_rows))
        for keys, entry_rows in sides:
            if len(keys) == 0:
                continue
            left = np.searchsorted(keys, qkeys, "left")
            right = np.searchsorted(keys, qkeys, "right")
            counts = right - left
            total = int(counts.sum())
            if total == 0:
                continue
            starts = np.cumsum(counts) - counts
            found_i.append(np.repeat(qrows, counts))
            found_j.append(entry_rows[np.repeat(left - starts, counts) + np.arange(total)])
        if not found_i:
            return empty, empty
        ii = np.concatenate(found_i)
        jj = np.concatenate(found_j)
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
        a = np.minimum(ii, jj)
        b = np.maximum(ii, jj)
        packed = np.unique(a * self._n + b)
        a = packed // self._n
        b = packed % self._n
        if ignore_static:
            keep = (self.mode[a] != MODE_STATIC) & (self.mode[b] != MODE_STATIC)
            a, b = a[keep], b[keep]
        margin = np.broadcast_to(np.asarray(clearance, dtype=float), (len(a), 3)).copy()
        two_d = (self.mode[a] == MODE_NORMAL_2D) | (self.mode[b] == MODE_NORMAL_2D)
        margin[two_d, 2] = 0.0
        if hybrid:
            margin[self.physics[a] | self.physics[b]] = 0.0
        overlap = ((self.aabb_max[a] + margin >= self.aabb_min[b]) & (self.aabb_min[a] - margin <= self.aabb_max[b])).all(
            axis=1
        )
        a, b = a[overlap], b[overlap]
        return self.object_id[a], self.object_id[b]
