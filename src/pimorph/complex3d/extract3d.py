"""Exact extraction of a 3-D cell complex from a voxel label volume.

Algorithm (crack complex on the doubled interpixel grid, shape (2Z+5, 2Y+5, 2X+5)
after a one-voxel padding shell of outer background):

1. 3-cells: every label is split into 6-connected components (cells). Background
   components touching the volume border merge into the single outer 3-cell;
   enclosed background components become gap 3-cells.
2. Voxel faces between voxels of different 3-cell ids are "on". An on-face has
   exactly one even doubled-grid coordinate (its normal axis).
3. A voxel edge is a skeleton edge when it is surrounded by >= 3 on-faces
   (>= 3 distinct ids, or the checkerboard A B A B). Edges with exactly 2 on-faces
   are manifold edges interior to an interface.
4. 2-cells (interfaces) are the connected components of on-faces linked through
   manifold edges. Every interface separates exactly two 3-cell ids; its normal
   points from the smaller id to the larger one.
5. 0-cells are corners whose skeleton degree is not 0 or 2 (branch points, and
   dead ends of checkerboard edges). Removing them splits the skeleton edges into
   chains, one per 1-cell (triple line). A closed chain without a 0-cell gets one
   artificial degree-2 vertex, as in 2-D.
6. An interface without any skeleton edge (a closed surface, e.g. a cell fully
   inside another) gets one artificial 1-cell (a single voxel edge of the surface)
   with two artificial 0-cells at its ends. Its boundary chain is zero, so the
   2-cell counts as a sphere minus an arc (an open disk) in Euler counts.

Boundary coefficients of interfaces on 1-cells are accumulated geometrically from
the voxel faces (right-hand rule with the interface normal), so B1 @ B2 == 0 and
B2 @ B3 == 0 hold identically for any input.

Coordinates are (z, row, col); doubled-grid coordinate d maps to d / 2 - 1.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label


class CellKind3D(IntEnum):
    CELL = 0
    GAP = 1
    OUTER = 2


class VertexKind3D(IntEnum):
    REGULAR = 0
    ARTIFICIAL = 1


class LineKind3D(IntEnum):
    REGULAR = 0
    ARTIFICIAL = 1


_LEVI = np.zeros((3, 3, 3), dtype=np.int8)
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _LEVI[_i, _j, _k] = 1
    _LEVI[_i, _k, _j] = -1


def _trace_chains_py(
    chain_id: np.ndarray, starts: np.ndarray, sizes: np.ndarray, is_cycle: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Walk each chain in the doubled grid. Returns the flat sequence of visited
    positions (start corner, then alternating edge/corner, then closing corner)
    and per-chain offsets. Paths start at their tail vertex corner, cycles at the
    artificial vertex."""
    n = starts.shape[0]
    offsets = np.zeros(n + 1, dtype=np.int64)
    for c in range(n):
        offsets[c + 1] = offsets[c] + sizes[c] + 2
    seq = np.empty((offsets[n], 3), dtype=np.int64)
    Z, Y, X = chain_id.shape
    for c in range(n):
        cid = c + 1
        base = offsets[c]
        z, y, x = int(starts[c, 0]), int(starts[c, 1]), int(starts[c, 2])
        pz, py, px = -1, -1, -1
        seq[base, 0] = z
        seq[base, 1] = y
        seq[base, 2] = x
        for k in range(int(sizes[c])):
            nz, ny, nx = -1, -1, -1
            if z > 0 and chain_id[z - 1, y, x] == cid and not (z - 1 == pz and y == py and x == px):
                nz, ny, nx = z - 1, y, x
            elif z + 1 < Z and chain_id[z + 1, y, x] == cid and not (z + 1 == pz and y == py and x == px):
                nz, ny, nx = z + 1, y, x
            elif y > 0 and chain_id[z, y - 1, x] == cid and not (z == pz and y - 1 == py and x == px):
                nz, ny, nx = z, y - 1, x
            elif y + 1 < Y and chain_id[z, y + 1, x] == cid and not (z == pz and y + 1 == py and x == px):
                nz, ny, nx = z, y + 1, x
            elif x > 0 and chain_id[z, y, x - 1] == cid and not (z == pz and y == py and x - 1 == px):
                nz, ny, nx = z, y, x - 1
            elif x + 1 < X and chain_id[z, y, x + 1] == cid and not (z == pz and y == py and x + 1 == px):
                nz, ny, nx = z, y, x + 1
            if nz < 0:
                break
            pz, py, px = z, y, x
            z, y, x = nz, ny, nx
            seq[base + k + 1, 0] = z
            seq[base + k + 1, 1] = y
            seq[base + k + 1, 2] = x
        last = base + sizes[c] + 1
        if is_cycle[c]:
            seq[last, 0] = starts[c, 0]
            seq[last, 1] = starts[c, 1]
            seq[last, 2] = starts[c, 2]
        else:
            seq[last, 0] = 2 * z - pz
            seq[last, 1] = 2 * y - py
            seq[last, 2] = 2 * x - px
    return seq, offsets


try:  # optional acceleration
    import numba as _nb

    _trace_chains = _nb.njit(cache=True)(_trace_chains_py)
except Exception:  # pragma: no cover
    _trace_chains = _trace_chains_py


def _count6(mask: np.ndarray) -> np.ndarray:
    """Number of True 6-neighbours of every doubled-grid site."""
    out = np.zeros(mask.shape, dtype=np.int8)
    out[1:] += mask[:-1]
    out[:-1] += mask[1:]
    out[:, 1:] += mask[:, :-1]
    out[:, :-1] += mask[:, 1:]
    out[:, :, 1:] += mask[:, :, :-1]
    out[:, :, :-1] += mask[:, :, 1:]
    return out


def _parity_count(shape: Tuple[int, int, int]) -> np.ndarray:
    """Number of odd coordinates per site: 3 voxel, 2 face, 1 edge, 0 corner."""
    z = (np.arange(shape[0]) % 2).astype(np.int8)
    y = (np.arange(shape[1]) % 2).astype(np.int8)
    x = (np.arange(shape[2]) % 2).astype(np.int8)
    return z[:, None, None] + y[None, :, None] + x[None, None, :]


def dg_to_xyz(dg: np.ndarray) -> np.ndarray:
    """Doubled-grid coordinates to voxel coordinates (voxel (z, y, x) centred at (z, y, x))."""
    return np.asarray(dg, dtype=np.float64) / 2.0 - 1.5


def _cell_map_from_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Padded 3-cell id map plus cell_kind, cell_label, cell_voxels."""
    cells = cc_label(labels, connectivity=1, background=0)
    n_cells = int(cells.max())
    cell_label = np.zeros(n_cells, dtype=np.int64)
    if n_cells:
        flat_c = cells.ravel()
        flat_l = labels.ravel()
        idx = np.flatnonzero(flat_c)
        lookup = np.zeros(n_cells + 1, dtype=np.int64)
        lookup[flat_c[idx]] = flat_l[idx]
        cell_label = lookup[1:]
    n_split = int(n_cells - np.count_nonzero(np.bincount(labels.ravel())[1:])) if n_cells else 0

    bg = cc_label(labels == 0, connectivity=1)
    n_bg = int(bg.max())
    border_touch = np.zeros(n_bg + 1, dtype=bool)
    if n_bg:
        for side in (bg[0], bg[-1], bg[:, 0], bg[:, -1], bg[:, :, 0], bg[:, :, -1]):
            border_touch[np.unique(side)] = True
    border_touch[0] = False
    enclosed_ids = np.flatnonzero(~border_touch[1:]) + 1
    n_gaps = int(enclosed_ids.size)
    outer_id = n_cells + n_gaps

    cell_map = np.full(tuple(s + 2 for s in labels.shape), outer_id, dtype=np.int32)
    inner = cell_map[1:-1, 1:-1, 1:-1]
    inner[cells > 0] = cells[cells > 0] - 1
    if n_gaps:
        gap_index = np.full(n_bg + 1, -1, dtype=np.int64)
        gap_index[enclosed_ids] = np.arange(n_gaps) + n_cells
        m = (labels == 0) & (gap_index[bg] >= 0)
        inner[m] = gap_index[bg[m]]

    cell_kind = np.concatenate(
        [
            np.full(n_cells, CellKind3D.CELL, dtype=np.int8),
            np.full(n_gaps, CellKind3D.GAP, dtype=np.int8),
            np.array([CellKind3D.OUTER], dtype=np.int8),
        ]
    )
    cell_label_full = np.concatenate([cell_label, np.full(n_gaps + 1, -1, dtype=np.int64)])
    cell_voxels = np.bincount(inner.ravel(), minlength=outer_id + 1).astype(np.int64)
    info = {
        "n_cells": n_cells,
        "n_gaps": n_gaps,
        "n_split_labels": n_split,
        "n_border_background_components": int(border_touch.sum()),
    }
    return cell_map, cell_kind, cell_label_full, cell_voxels, info


@dataclass
class CellComplex3D:
    """A 3-D cell complex extracted from a label volume.

    Arrays are indexed by vertex (0-cell), line (1-cell), interface (2-cell) and
    cell (3-cell) ids. ``interface_cells[f] = (a, b)`` with a < b; the interface
    normal points from a into b. ``interface_boundary[f]`` holds rows
    ``(line, coefficient)`` of the boundary chain; ``interface_loops[f]`` holds one
    Eulerian circuit of signed half-lines ``2 * e + d`` (d = 0 means the line is
    traversed tail -> head) per connected component of the boundary, including
    slit lines whose coefficient cancels. ``face_dg`` lists the doubled-grid coordinates
    of all voxel faces, grouped by interface through ``interface_face_ptr``.
    """

    cell_kind: np.ndarray  # (C,) int8
    cell_label: np.ndarray  # (C,) int64, -1 for gap/outer
    cell_voxels: np.ndarray  # (C,) int64
    interface_cells: np.ndarray  # (F, 2) int64
    interface_area: np.ndarray  # (F,) int64 voxel faces
    interface_centroid: np.ndarray  # (F, 3) float64 (z, y, x)
    face_dg: np.ndarray  # (N, 3) int32 doubled-grid coordinates
    interface_face_ptr: np.ndarray  # (F + 1,) int64
    interface_boundary: List[np.ndarray]  # F arrays (m, 2) int64 (line, coef)
    interface_loops: List[List[List[int]]]
    line_tail: np.ndarray  # (E,) int64
    line_head: np.ndarray  # (E,) int64
    line_length: np.ndarray  # (E,) int64 voxel edges
    line_kind: np.ndarray  # (E,) int8
    line_polyline: List[np.ndarray]  # E arrays (n_k + 1, 3) corner positions tail -> head
    line_interfaces: List[np.ndarray]  # E arrays of geometrically incident interface ids
    vertex_xyz: np.ndarray  # (V, 3) float64
    vertex_kind: np.ndarray  # (V,) int8
    shape: Tuple[int, int, int]
    cell_map: np.ndarray  # (Z + 2, Y + 2, X + 2) int32 3-cell id per voxel, padded with the outer id
    provenance: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ sizes
    @property
    def n_vertices(self) -> int:
        return int(self.vertex_xyz.shape[0])

    @property
    def n_lines(self) -> int:
        return int(self.line_tail.shape[0])

    @property
    def n_interfaces(self) -> int:
        return int(self.interface_cells.shape[0])

    @property
    def n_cells(self) -> int:
        """Number of 3-cells including gaps and the outer cell."""
        return int(self.cell_kind.shape[0])

    # ------------------------------------------------------------------ cells
    def cells_of_kind(self, kind: CellKind3D) -> np.ndarray:
        return np.flatnonzero(self.cell_kind == int(kind))

    @property
    def cell_cells(self) -> np.ndarray:
        return self.cells_of_kind(CellKind3D.CELL)

    @property
    def gap_cells(self) -> np.ndarray:
        return self.cells_of_kind(CellKind3D.GAP)

    @property
    def outer_cell(self) -> int:
        idx = self.cells_of_kind(CellKind3D.OUTER)
        if idx.size != 1:
            raise ValueError(f"expected exactly one outer cell, found {idx.size}")
        return int(idx[0])

    def interfaces_of_cell(self, c: int) -> np.ndarray:
        return np.flatnonzero((self.interface_cells[:, 0] == c) | (self.interface_cells[:, 1] == c))

    def cell_neighbors(self, c: int, kinds: Sequence[CellKind3D] = (CellKind3D.CELL,)) -> np.ndarray:
        """Distinct 3-cells sharing an interface with c, filtered by kind."""
        want = {int(k) for k in kinds}
        f = self.interfaces_of_cell(c)
        other = np.where(self.interface_cells[f, 0] == c, self.interface_cells[f, 1], self.interface_cells[f, 0])
        other = np.unique(other)
        return other[np.isin(self.cell_kind[other], list(want))]

    def cell_touches_outer(self, c: int) -> bool:
        return bool(np.any(self.interface_cells[self.interfaces_of_cell(c)] == self.outer_cell))

    # ------------------------------------------------------------- interfaces
    def interface_faces(self, f: int) -> np.ndarray:
        """(n, 4) int rows (axis, z, y, x): normal axis and doubled-grid coordinates."""
        dg = self.face_dg[self.interface_face_ptr[f] : self.interface_face_ptr[f + 1]]
        axis = np.argmin(dg % 2, axis=1).astype(np.int32)
        return np.concatenate([axis[:, None], dg], axis=1)

    def interface_lines(self, f: int) -> np.ndarray:
        """Distinct 1-cells on the boundary loops of interface f."""
        return np.unique(np.array([h >> 1 for loop in self.interface_loops[f] for h in loop], dtype=np.int64))

    def interface_vertices(self, f: int) -> np.ndarray:
        lines = self.interface_lines(f)
        return np.unique(np.concatenate([self.line_tail[lines], self.line_head[lines]]))

    def interfaces_between(self, a: int, b: int) -> np.ndarray:
        lo, hi = min(a, b), max(a, b)
        return np.flatnonzero((self.interface_cells[:, 0] == lo) & (self.interface_cells[:, 1] == hi))

    def interface_is_boundary(self, f: int) -> bool:
        return bool(self.interface_cells[f, 1] == self.outer_cell)

    # ------------------------------------------------------------------ lines
    def vertex_degrees(self) -> np.ndarray:
        deg = np.zeros(self.n_vertices, dtype=np.int64)
        np.add.at(deg, self.line_tail, 1)
        np.add.at(deg, self.line_head, 1)
        return deg

    def line_cells(self, e: int) -> np.ndarray:
        """Distinct 3-cells around a 1-cell, from its incident interfaces."""
        f = self.line_interfaces[e]
        return np.unique(self.interface_cells[f].ravel()) if f.size else np.zeros(0, dtype=np.int64)


def _euler_circuits(traversals: List[Tuple[int, int]], tail: np.ndarray, head: np.ndarray) -> List[List[int]]:
    """Decompose a multiset of directed line traversals (line, d) into one Eulerian
    circuit per connected component (Hierholzer). The multiset is balanced (the
    boundary of a boundary is zero), so every component is Eulerian. Returns lists
    of half-line ids ``2 * e + d``."""
    n = len(traversals)
    starts = [int(tail[e]) if d == 0 else int(head[e]) for e, d in traversals]
    ends = [int(head[e]) if d == 0 else int(tail[e]) for e, d in traversals]
    out_at: Dict[int, List[int]] = {}
    for i, v in enumerate(starts):
        out_at.setdefault(v, []).append(i)
    ptr: Dict[int, int] = {v: 0 for v in out_at}
    used = [False] * n
    circuits: List[List[int]] = []
    for i0 in range(n):
        if used[i0]:
            continue
        stack: List[Tuple[int, int]] = [(starts[i0], -1)]
        circuit: List[int] = []
        while stack:
            v, h_in = stack[-1]
            lst = out_at.get(v, [])
            while ptr[v] < len(lst) and used[lst[ptr[v]]]:
                ptr[v] += 1
            if ptr[v] < len(lst):
                i = lst[ptr[v]]
                used[i] = True
                ptr[v] += 1
                stack.append((ends[i], i))
            else:
                stack.pop()
                if h_in >= 0:
                    circuit.append(h_in)
        circuit.reverse()
        circuits.append([2 * traversals[i][0] + traversals[i][1] for i in circuit])
    return circuits


def extract_complex3d(labels: np.ndarray, provenance: Dict | None = None) -> CellComplex3D:
    """Build the exact 3-D cell complex of an integer label volume (z, y, x).

    Background is 0. The returned complex always has exactly one outer 3-cell and
    exact boundary operators (see module docstring).
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"labels must be 3-D, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must be an integer array")
    if labels.size and labels.min() < 0:
        raise ValueError("negative labels are not allowed")
    shape = tuple(int(s) for s in labels.shape)

    cell_map, cell_kind, cell_label, cell_voxels, info = _cell_map_from_labels(labels)
    Zp, Yp, Xp = cell_map.shape
    dg_shape = (2 * Zp + 1, 2 * Yp + 1, 2 * Xp + 1)

    # ---------------------------------------------------------------- faces
    face_dg_parts, face_lo_parts, face_hi_parts = [], [], []
    for a in range(3):
        lo = np.take(cell_map, np.arange(cell_map.shape[a] - 1), axis=a)
        hi = np.take(cell_map, np.arange(1, cell_map.shape[a]), axis=a)
        on = lo != hi
        idx = np.nonzero(on)
        dg = np.stack([2 * i + 1 for i in idx], axis=1).astype(np.int32)
        dg[:, a] += 1
        face_dg_parts.append(dg)
        face_lo_parts.append(lo[idx].astype(np.int64))
        face_hi_parts.append(hi[idx].astype(np.int64))
    face_dg_all = np.concatenate(face_dg_parts, axis=0)
    face_lo = np.concatenate(face_lo_parts)
    face_hi = np.concatenate(face_hi_parts)
    n_faces_total = face_dg_all.shape[0]

    F_on = np.zeros(dg_shape, dtype=bool)
    F_on[face_dg_all[:, 0], face_dg_all[:, 1], face_dg_all[:, 2]] = True
    par = _parity_count(dg_shape)
    edge_pos = par == 1
    corner_pos = par == 0
    n_on = _count6(F_on)
    S = edge_pos & (n_on >= 3)  # skeleton edges
    M = edge_pos & (n_on == 2)  # manifold edges
    del n_on
    deg = _count6(S)
    vertex_corner = corner_pos & ((deg >= 3) | (deg == 1))
    chain_corner = corner_pos & (deg == 2)
    del deg, par, edge_pos, corner_pos

    # Interfaces: on-faces linked through manifold edges.
    comp, n_if = ndi.label(F_on | M)
    comp = comp.astype(np.int32)
    del M
    face_if = comp[face_dg_all[:, 0], face_dg_all[:, 1], face_dg_all[:, 2]].astype(np.int64) - 1
    del comp
    order = np.lexsort((face_dg_all[:, 2], face_dg_all[:, 1], face_dg_all[:, 0], face_if))
    face_dg_all = face_dg_all[order]
    face_if = face_if[order]
    face_lo = face_lo[order]
    face_hi = face_hi[order]
    face_area = np.bincount(face_if, minlength=n_if).astype(np.int64)
    face_ptr = np.zeros(n_if + 1, dtype=np.int64)
    face_ptr[1:] = np.cumsum(face_area)
    interface_cells = np.zeros((n_if, 2), dtype=np.int64)
    if n_if:
        first = face_ptr[:-1]
        interface_cells[:, 0] = np.minimum(face_lo[first], face_hi[first])
        interface_cells[:, 1] = np.maximum(face_lo[first], face_hi[first])
    face_nsign = np.where(face_lo < face_hi, 1, -1).astype(np.int8)  # normal from min id to max id
    centroid = np.zeros((n_if, 3), dtype=np.float64)
    if n_if:
        xyz = dg_to_xyz(face_dg_all)
        for k in range(3):
            centroid[:, k] = np.bincount(face_if, weights=xyz[:, k], minlength=n_if) / np.maximum(face_area, 1)
    if_dg = np.zeros(dg_shape, dtype=np.int32)
    if_dg[face_dg_all[:, 0], face_dg_all[:, 1], face_dg_all[:, 2]] = face_if + 1
    nsign_dg = np.zeros(dg_shape, dtype=np.int8)
    nsign_dg[face_dg_all[:, 0], face_dg_all[:, 1], face_dg_all[:, 2]] = face_nsign
    del F_on

    # ---------------------------------------------------------------- chains
    chain, n_chains = ndi.label(S | chain_corner)
    chain = chain.astype(np.int32)
    sizes = np.bincount(chain.ravel(), minlength=n_chains + 1)[1:].astype(np.int64)
    starts = np.full((n_chains, 3), -1, dtype=np.int64)
    is_cycle = np.zeros(n_chains, dtype=bool)
    if n_chains:
        ez, ey, ex = np.nonzero(S)
        e_pos = np.stack([ez, ey, ex], axis=1).astype(np.int64)
        e_axis = np.argmax(e_pos % 2, axis=1)
        step = np.zeros_like(e_pos)
        step[np.arange(e_pos.shape[0]), e_axis] = 1
        c_minus = e_pos - step
        c_plus = e_pos + step
        v_minus = vertex_corner[c_minus[:, 0], c_minus[:, 1], c_minus[:, 2]]
        v_plus = vertex_corner[c_plus[:, 0], c_plus[:, 1], c_plus[:, 2]]
        endpoint = v_minus | v_plus
        e_chain = chain[ez, ey, ex].astype(np.int64) - 1
        sel = np.flatnonzero(endpoint)
        cid_s, first_pos = np.unique(e_chain[sel], return_index=True)
        pick = sel[first_pos]
        starts[cid_s] = np.where(v_minus[pick][:, None], c_minus[pick], c_plus[pick])
        cyc = np.flatnonzero(starts[:, 0] < 0)
        if cyc.size:
            is_cycle[cyc] = True
            cz, cy, cx = np.nonzero(chain_corner)
            c_chain = chain[cz, cy, cx].astype(np.int64) - 1
            keep = np.isin(c_chain, cyc)
            cz, cy, cx, c_chain = cz[keep], cy[keep], cx[keep], c_chain[keep]
            cid_c, first_c = np.unique(c_chain, return_index=True)
            starts[cid_c, 0] = cz[first_c]
            starts[cid_c, 1] = cy[first_c]
            starts[cid_c, 2] = cx[first_c]
    del S, chain_corner

    seq, offsets = _trace_chains(chain, starts, sizes, is_cycle)
    del chain
    line_length = (sizes + 1) // 2

    # Per-element bookkeeping over the flat sequence.
    n_el = int(offsets[-1])
    el_chain = np.repeat(np.arange(n_chains, dtype=np.int64), sizes + 2)
    local = np.arange(n_el, dtype=np.int64) - offsets[el_chain]
    is_edge_el = (local % 2 == 1) & (local <= sizes[el_chain])
    edge_idx = np.flatnonzero(is_edge_el)
    e_dg = seq[edge_idx]
    delta = seq[edge_idx + 1] - seq[edge_idx - 1]  # +-2 along the edge axis
    e_dir = np.sign(delta.sum(axis=1)).astype(np.int8)
    line_dg = np.zeros(dg_shape, dtype=np.int32)
    dir_dg = np.zeros(dg_shape, dtype=np.int8)
    line_dg[e_dg[:, 0], e_dg[:, 1], e_dg[:, 2]] = el_chain[edge_idx].astype(np.int32) + 1
    dir_dg[e_dg[:, 0], e_dg[:, 1], e_dg[:, 2]] = e_dir

    # Vertices: regular corners first, then artificial cycle vertices.
    vz, vy, vx = np.nonzero(vertex_corner)
    del vertex_corner
    reg_dg = np.stack([vz, vy, vx], axis=1).astype(np.int64)
    art_dg = starts[is_cycle]
    vid_dg = np.full(dg_shape, -1, dtype=np.int32)
    vid_dg[reg_dg[:, 0], reg_dg[:, 1], reg_dg[:, 2]] = np.arange(reg_dg.shape[0], dtype=np.int32)
    vid_dg[art_dg[:, 0], art_dg[:, 1], art_dg[:, 2]] = np.arange(art_dg.shape[0], dtype=np.int32) + reg_dg.shape[0]
    vertex_dg_list = [reg_dg, art_dg]
    vertex_kind_list = [
        np.full(reg_dg.shape[0], VertexKind3D.REGULAR, dtype=np.int8),
        np.full(art_dg.shape[0], VertexKind3D.ARTIFICIAL, dtype=np.int8),
    ]
    tail_dg = seq[offsets[:-1]]
    head_dg = seq[offsets[1:] - 1]
    line_tail = vid_dg[tail_dg[:, 0], tail_dg[:, 1], tail_dg[:, 2]].astype(np.int64)
    line_head = vid_dg[head_dg[:, 0], head_dg[:, 1], head_dg[:, 2]].astype(np.int64)
    if n_chains and (line_tail.min() < 0 or line_head.min() < 0):
        raise RuntimeError("chain endpoint is not a vertex corner")
    del vid_dg
    line_polyline: List[np.ndarray] = []
    corner_sel = local % 2 == 0
    for c in range(n_chains):
        pts = seq[offsets[c] : offsets[c + 1]]
        line_polyline.append(dg_to_xyz(pts[corner_sel[offsets[c] : offsets[c + 1]]]))
    line_kind_list = [np.full(n_chains, LineKind3D.REGULAR, dtype=np.int8)]

    # ------------------------------------------- boundary incidences (fine level)
    key_parts, sign_parts = [], []
    for a in range(3):
        sel = np.argmin(face_dg_all % 2, axis=1) == a
        fdg = face_dg_all[sel].astype(np.int64)
        fif = face_if[sel]
        fns = face_nsign[sel].astype(np.int64)
        for c in range(3):
            if c == a:
                continue
            b = 3 - a - c
            levi = int(_LEVI[c, a, b])
            for s in (-1, 1):
                edg = fdg.copy()
                edg[:, c] += s
                lid = line_dg[edg[:, 0], edg[:, 1], edg[:, 2]].astype(np.int64)
                m = lid > 0
                if not m.any():
                    continue
                d = dir_dg[edg[m, 0], edg[m, 1], edg[m, 2]].astype(np.int64)
                sign = -s * fns[m] * levi * d
                key_parts.append((lid[m] - 1) * max(n_if, 1) + fif[m])
                sign_parts.append(sign)
    del line_dg, dir_dg, if_dg, nsign_dg
    if key_parts:
        keys = np.concatenate(key_parts)
        signs = np.concatenate(sign_parts)
        ukeys, inv = np.unique(keys, return_inverse=True)
        n_fwd = np.bincount(inv, weights=(signs > 0).astype(np.float64)).astype(np.int64)
        n_bwd = np.bincount(inv, weights=(signs < 0).astype(np.float64)).astype(np.int64)
        k_line = ukeys // max(n_if, 1)
        k_if = ukeys % max(n_if, 1)
    else:
        k_line = np.zeros(0, dtype=np.int64)
        k_if = np.zeros(0, dtype=np.int64)
        n_fwd = n_bwd = np.zeros(0, dtype=np.int64)
    net = n_fwd - n_bwd
    if k_line.size and np.any(net % line_length[k_line] != 0):
        raise RuntimeError("boundary coefficient varies along a 1-cell")
    # Whole-line traversal multiplicities. A turn-back in the middle of a line (a
    # slit that does not reach a 0-cell) leaves a remainder and is dropped.
    if k_line.size:
        coef = net // line_length[k_line]
        m_fwd = n_fwd // line_length[k_line]
        m_bwd = n_bwd // line_length[k_line]
    else:
        coef = m_fwd = m_bwd = net

    line_interfaces: List[List[int]] = [[] for _ in range(n_chains)]
    boundary_rows: List[List[Tuple[int, int]]] = [[] for _ in range(n_if)]
    traversal_rows: List[List[Tuple[int, int]]] = [[] for _ in range(n_if)]
    for e, f, cf, mf, mb in zip(k_line.tolist(), k_if.tolist(), coef.tolist(), m_fwd.tolist(), m_bwd.tolist()):
        line_interfaces[e].append(f)
        if cf != 0:
            boundary_rows[f].append((e, cf))
        traversal_rows[f].extend([(e, 0)] * mf + [(e, 1)] * mb)

    # ----------------------------- artificial arcs on closed interfaces
    has_skeleton = np.zeros(n_if, dtype=bool)
    if k_if.size:
        has_skeleton[np.unique(k_if)] = True
    closed = np.flatnonzero(~has_skeleton)
    n_art_lines = int(closed.size)
    art_tail = np.zeros((n_art_lines, 3), dtype=np.int64)
    art_head = np.zeros((n_art_lines, 3), dtype=np.int64)
    art_loops: Dict[int, List[List[int]]] = {}
    n_v_before = reg_dg.shape[0] + art_dg.shape[0]
    for k, f in enumerate(closed.tolist()):
        g = face_dg_all[face_ptr[f]].astype(np.int64)
        a = int(np.argmin(g % 2))
        c1, c2 = [ax for ax in range(3) if ax != a]
        # edge on the low side along c1, running along c2
        t = g.copy()
        t[c1] -= 1
        t[c2] -= 1
        h = t.copy()
        h[c2] += 2
        art_tail[k] = t
        art_head[k] = h
        e = n_chains + k
        line_interfaces.append([f])
        line_polyline.append(dg_to_xyz(np.stack([t, h], axis=0)))
        art_loops[f] = [[2 * e, 2 * e + 1]]
    if n_art_lines:
        vertex_dg_list += [art_tail, art_head]
        vertex_kind_list.append(np.full(2 * n_art_lines, VertexKind3D.ARTIFICIAL, dtype=np.int8))
        line_tail = np.concatenate([line_tail, n_v_before + np.arange(n_art_lines)])
        line_head = np.concatenate([line_head, n_v_before + n_art_lines + np.arange(n_art_lines)])
        line_length = np.concatenate([line_length, np.ones(n_art_lines, dtype=np.int64)])
        line_kind_list.append(np.full(n_art_lines, LineKind3D.ARTIFICIAL, dtype=np.int8))

    vertex_dg = np.concatenate(vertex_dg_list, axis=0) if vertex_dg_list else np.zeros((0, 3), dtype=np.int64)
    vertex_kind = np.concatenate(vertex_kind_list) if vertex_kind_list else np.zeros(0, dtype=np.int8)
    line_kind = np.concatenate(line_kind_list) if line_kind_list else np.zeros(0, dtype=np.int8)

    interface_boundary: List[np.ndarray] = []
    interface_loops: List[List[List[int]]] = []
    for f in range(n_if):
        rows = boundary_rows[f]
        interface_boundary.append(np.array(rows, dtype=np.int64).reshape(-1, 2))
        if f in art_loops:
            interface_loops.append(art_loops[f])
            continue
        trav = traversal_rows[f]
        interface_loops.append(_euler_circuits(trav, line_tail, line_head) if trav else [])

    prov = {**info, **(provenance or {}), "connectivity": 6, "n_voxel_faces": int(n_faces_total)}
    return CellComplex3D(
        cell_kind=cell_kind,
        cell_label=cell_label,
        cell_voxels=cell_voxels,
        interface_cells=interface_cells,
        interface_area=face_area,
        interface_centroid=centroid,
        face_dg=face_dg_all,
        interface_face_ptr=face_ptr,
        interface_boundary=interface_boundary,
        interface_loops=interface_loops,
        line_tail=line_tail.astype(np.int64),
        line_head=line_head.astype(np.int64),
        line_length=line_length.astype(np.int64),
        line_kind=line_kind,
        line_polyline=line_polyline,
        line_interfaces=[np.array(sorted(set(v)), dtype=np.int64) for v in line_interfaces],
        vertex_xyz=dg_to_xyz(vertex_dg).reshape(-1, 3),
        vertex_kind=vertex_kind,
        shape=shape,  # type: ignore[arg-type]
        cell_map=cell_map,
        provenance=prov,
    )
