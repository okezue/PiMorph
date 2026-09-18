"""Exact topological identities of the 3-D complex used as reconstruction QC."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .extract3d import CellComplex3D, CellKind3D


def euler_characteristic3d(cx: CellComplex3D, include_outer: bool = True) -> int:
    """V - E + F - C. With the outer cell this is 0 for a connected cellulation of
    the 3-ball (chi of S^3); without it, 1."""
    C = cx.n_cells if include_outer else cx.n_cells - 1
    return int(cx.n_vertices - cx.n_lines + cx.n_interfaces - C)


def per_cell_euler(cx: CellComplex3D, c: int, exact: bool = False, loop_corrected: bool = True) -> int:
    """Euler characteristic of the boundary of 3-cell c.

    Default: V - E + F counted over the coarse complex (interfaces of c, the 1-cells
    on their boundary circuits and the 0-cells at their ends), minus one per extra
    boundary component of an interface (an interface with L boundary components is
    a planar surface with chi = 2 - L, not a disk). Equals 2 for a ball-like cell
    with planar interfaces, 2 + 2m with m cavities. Closed interfaces carry a slit
    and count as spheres, so handles are invisible here. ``exact=True`` counts the
    voxel boundary surface itself (``boundary_surface_euler``): 2 for a ball, 0 for
    a solid torus. ``loop_corrected=False`` gives the raw V - E + F.
    """
    if exact:
        return boundary_surface_euler(cx.cell_map == c)
    faces = cx.interfaces_of_cell(c)
    if faces.size == 0:
        return 0
    lines = np.unique(np.concatenate([cx.interface_lines(int(f)) for f in faces]))
    if lines.size:
        verts = np.unique(np.concatenate([cx.line_tail[lines], cx.line_head[lines]]))
    else:
        verts = np.zeros(0, dtype=np.int64)
    chi = int(verts.size - lines.size + faces.size)
    if loop_corrected:
        chi -= int(sum(max(len(cx.interface_loops[int(f)]), 1) - 1 for f in faces))
    return chi


def _link_component_lut() -> np.ndarray:
    """For each 8-voxel membership pattern around a corner, the number of connected
    components of the boundary faces there. Faces are glued along edges with two
    boundary faces; at checkerboard edges (four boundary faces) the two faces of the
    same member voxel are glued, so a voxel set pinched along an edge or a corner has
    a separate boundary sheet for each part."""
    lut = np.zeros(256, dtype=np.int64)
    voxels = [(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    bit = {v: 4 * v[0] + 2 * v[1] + v[2] for v in voxels}
    faces = []
    for i, u in enumerate(voxels):
        for v in voxels[i + 1 :]:
            if sum(abs(a - b) for a, b in zip(u, v)) == 1:
                faces.append((u, v))
    for pat in range(256):
        inside = {v: bool(pat >> bit[v] & 1) for v in voxels}
        bfaces = [(u, v) for u, v in faces if inside[u] != inside[v]]
        parent = list(range(len(bfaces)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for axis in range(3):
            for side in (0, 1):
                around = [k for k, (u, v) in enumerate(bfaces) if u[axis] == side and v[axis] == side]
                if len(around) == 2:
                    parent[find(around[0])] = find(around[1])
                elif len(around) == 4:
                    for k in around:
                        for m in around:
                            if k < m:
                                u1, v1 = bfaces[k]
                                u2, v2 = bfaces[m]
                                shared = {w for w in (u1, v1) if inside[w]} & {w for w in (u2, v2) if inside[w]}
                                if shared:
                                    parent[find(k)] = find(m)
        lut[pat] = len({find(k) for k in range(len(bfaces))})
    return lut


_LUT = _link_component_lut()


def boundary_surface_euler(mask: np.ndarray) -> int:
    """Exact Euler characteristic of the boundary surface of a voxel set.

    The surface is the boundary of the open 6-connected region: faces between member
    and non-member voxels, glued along edges shared by two such faces (or, at a
    checkerboard edge, by the same member voxel). Every abstract edge has two faces,
    so chi = V_abstract - F with V_abstract from ``_link_component_lut``. 2 for a
    ball, 0 for a solid torus, 2 + 2m for a ball with m cavities."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    m = np.pad(mask, 1)
    pat = np.zeros(tuple(s - 1 for s in m.shape), dtype=np.int64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                sl = (slice(dz, m.shape[0] - 1 + dz), slice(dy, m.shape[1] - 1 + dy), slice(dx, m.shape[2] - 1 + dx))
                pat |= m[sl].astype(np.int64) << (4 * dz + 2 * dy + dx)
    v_abs = int(_LUT[pat].sum())
    n_faces = 0
    for a in range(3):
        lo = np.take(m, np.arange(m.shape[a] - 1), axis=a)
        hi = np.take(m, np.arange(1, m.shape[a]), axis=a)
        n_faces += int(np.count_nonzero(lo != hi))
    return v_abs - n_faces


def neighbor_counts(cx: CellComplex3D, kinds: Sequence[CellKind3D] = (CellKind3D.CELL,)) -> np.ndarray:
    """(C,) number of distinct neighbouring 3-cells of the given kinds per 3-cell.
    Two cells sharing several interfaces (multiple contacts) count once."""
    out = np.zeros(cx.n_cells, dtype=np.int64)
    for c in range(cx.n_cells):
        out[c] = cx.cell_neighbors(c, kinds=kinds).size
    return out


def contact_multiplicity(cx: CellComplex3D) -> np.ndarray:
    """(F,) number of distinct interfaces shared by the same pair of 3-cells as f."""
    pair = cx.interface_cells[:, 0] * cx.n_cells + cx.interface_cells[:, 1]
    _, inv, cnt = np.unique(pair, return_inverse=True, return_counts=True)
    return cnt[inv].astype(np.int64)
