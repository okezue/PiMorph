"""Exact extraction of a half-edge cell complex from a label image.

Algorithm (crack graph on the doubled interpixel grid):

1. Split every label into 4-connected components (faces must be simply connected
   regions; a pinch corner therefore yields two faces and a degree-4 vertex).
2. Background components that touch the image border are merged into the single
   outer face; enclosed background components become gap faces.
3. Cracks are placed between 4-adjacent pixels whose face ids differ. Crack corners
   with 3 or 4 incident cracks are vertices. Removing the vertex corners splits the
   crack set into chains, one per edge. Closed chains with no vertex get one
   artificial degree-2 vertex.
4. Left/right faces of each edge come from the pixels flanking its first crack.
   ``he_next`` follows from the rotation system, and the traced loops are checked
   against the label faces.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label

from .halfedge import FaceKind, HalfEdgeComplex, VertexKind, build_rotation_system, next_from_rotation

_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _trace_all_py(chain_id: np.ndarray, starts: np.ndarray, sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = starts.shape[0]
    total = int(sizes.sum())
    coords = np.empty((total, 2), dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    H, W = chain_id.shape
    for c in range(n):
        cid = c + 1
        r, q = int(starts[c, 0]), int(starts[c, 1])
        pr, pq = -1, -1
        base = offsets[c]
        for k in range(int(sizes[c])):
            coords[base + k, 0] = r
            coords[base + k, 1] = q
            nr, nq = -1, -1
            if r > 0 and chain_id[r - 1, q] == cid and not (r - 1 == pr and q == pq):
                nr, nq = r - 1, q
            elif r + 1 < H and chain_id[r + 1, q] == cid and not (r + 1 == pr and q == pq):
                nr, nq = r + 1, q
            elif q > 0 and chain_id[r, q - 1] == cid and not (r == pr and q - 1 == pq):
                nr, nq = r, q - 1
            elif q + 1 < W and chain_id[r, q + 1] == cid and not (r == pr and q + 1 == pq):
                nr, nq = r, q + 1
            pr, pq = r, q
            if nr < 0:
                break
            r, q = nr, nq
    return coords, offsets


try:  # optional acceleration
    import numba as _nb

    _trace_all = _nb.njit(cache=True)(_trace_all_py)
except Exception:  # pragma: no cover
    _trace_all = _trace_all_py


def _face_map_from_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Return padded face-id map plus face_kind, face_label arrays."""
    H, W = labels.shape
    cells = cc_label(labels, connectivity=1, background=0)  # 0 stays 0
    n_cells = int(cells.max())
    face_label = np.zeros(n_cells, dtype=np.int64)
    if n_cells:
        # Any pixel of a component gives its original label; a scatter keeps the
        # last write, which is fine because all pixels of a component share one label.
        flat_c = cells.ravel()
        flat_l = labels.ravel()
        idx = np.flatnonzero(flat_c)
        lookup = np.zeros(n_cells + 1, dtype=np.int64)
        lookup[flat_c[idx]] = flat_l[idx]
        face_label = lookup[1:]
    n_split = int(n_cells - np.count_nonzero(np.bincount(labels.ravel())[1:])) if n_cells else 0

    bg = cc_label(labels == 0, connectivity=1)
    n_bg = int(bg.max())
    border_touch = np.zeros(n_bg + 1, dtype=bool)
    if n_bg:
        for edge in (bg[0, :], bg[-1, :], bg[:, 0], bg[:, -1]):
            border_touch[np.unique(edge)] = True
    border_touch[0] = False
    enclosed_ids = np.flatnonzero(~border_touch[1:]) + 1  # bg component ids that are gaps
    n_gaps = int(enclosed_ids.size)
    outer_id = n_cells + n_gaps

    face_map = np.full((H + 2, W + 2), outer_id, dtype=np.int64)
    inner = face_map[1:-1, 1:-1]
    inner[cells > 0] = cells[cells > 0] - 1
    if n_gaps:
        gap_index = np.full(n_bg + 1, -1, dtype=np.int64)
        gap_index[enclosed_ids] = np.arange(n_gaps) + n_cells
        m = (labels == 0) & (gap_index[bg] >= 0)
        inner[m] = gap_index[bg[m]]

    face_kind = np.concatenate(
        [
            np.full(n_cells, FaceKind.CELL, dtype=np.int8),
            np.full(n_gaps, FaceKind.GAP, dtype=np.int8),
            np.array([FaceKind.OUTER], dtype=np.int8),
        ]
    )
    face_label_full = np.concatenate([face_label, np.full(n_gaps + 1, -1, dtype=np.int64)])
    info = {
        "n_cells": n_cells,
        "n_gaps": n_gaps,
        "n_split_labels": n_split,
        "n_border_background_components": int(border_touch.sum()),
    }
    return face_map, face_kind, face_label_full, info


def extract_complex(
    labels: np.ndarray,
    pixel_size_um: Optional[float] = None,
    provenance: Optional[Dict] = None,
) -> HalfEdgeComplex:
    """Build the exact half-edge complex of a 2-D integer label image.

    Background is 0. The returned complex always contains exactly one outer face,
    and ``validate(cx).ok`` is True by construction for any input.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must be an integer array")
    if labels.min() < 0:
        raise ValueError("negative labels are not allowed")
    H, W = labels.shape

    face_map, face_kind, face_label, info = _face_map_from_labels(labels)
    Hp, Wp = face_map.shape

    # Cracks on the doubled grid. Corner (i, j) of the padded grid -> CG[2i, 2j].
    vcr = face_map[:, :-1] != face_map[:, 1:]  # (Hp, Wp-1): column line j+1, rows i..i+1
    hcr = face_map[:-1, :] != face_map[1:, :]  # (Hp-1, Wp): row line i+1, cols j..j+1
    CG = np.zeros((2 * Hp + 1, 2 * Wp + 1), dtype=bool)
    CG[1::2, 2:-1:2] = vcr  # rows 2i+1, cols 2(j+1)
    CG[2:-1:2, 1::2] = hcr  # rows 2(i+1), cols 2j+1

    # Corner degree from the four neighbouring crack cells.
    deg = np.zeros((Hp + 1, Wp + 1), dtype=np.int8)
    deg[1:, :] += CG[1::2, 0::2]  # crack above corner (i, j): CG[2i-1, 2j]
    deg[:-1, :] += CG[1::2, 0::2]  # crack below
    deg[:, 1:] += CG[0::2, 1::2]  # crack left
    deg[:, :-1] += CG[0::2, 1::2]  # crack right
    corner_on = deg >= 2
    vertex_corner = deg >= 3
    CG[0::2, 0::2] = corner_on

    chain_px = CG.copy()
    chain_px[0::2, 0::2] &= ~vertex_corner
    chain_id, n_chains = ndi.label(chain_px, structure=_FOUR)
    chain_id = chain_id.astype(np.int32)

    if n_chains == 0:
        return HalfEdgeComplex(
            vertex_xy=np.zeros((0, 2)),
            vertex_kind=np.zeros(0, dtype=np.int8),
            edge_tail=np.zeros(0, dtype=np.int64),
            edge_head=np.zeros(0, dtype=np.int64),
            edge_faces=np.zeros((0, 2), dtype=np.int64),
            edge_polyline=[],
            face_kind=face_kind,
            face_label=face_label,
            face_loops=[[] for _ in range(face_kind.size)],
            he_next=np.zeros(0, dtype=np.int64),
            pixel_size_um=pixel_size_um,
            shape=(H, W),
            provenance={**info, **(provenance or {}), "connectivity": 4},
        )

    # Chain degree (within the same chain) to find endpoints.
    same = np.zeros_like(chain_id, dtype=np.int8)
    same[1:, :] += (chain_id[1:, :] == chain_id[:-1, :]) & (chain_id[1:, :] > 0)
    same[:-1, :] += (chain_id[1:, :] == chain_id[:-1, :]) & (chain_id[:-1, :] > 0)
    same[:, 1:] += (chain_id[:, 1:] == chain_id[:, :-1]) & (chain_id[:, 1:] > 0)
    same[:, :-1] += (chain_id[:, 1:] == chain_id[:, :-1]) & (chain_id[:, :-1] > 0)

    sizes = np.bincount(chain_id.ravel(), minlength=n_chains + 1)[1:].astype(np.int64)
    starts = np.full((n_chains, 2), -1, dtype=np.int64)
    is_loop = np.zeros(n_chains, dtype=bool)

    ep_r, ep_c = np.nonzero((chain_id > 0) & (same <= 1))
    ep_ids = chain_id[ep_r, ep_c] - 1
    # first endpoint per chain (any is fine)
    order = np.argsort(ep_ids, kind="stable")
    ep_ids_s, first_pos = np.unique(ep_ids[order], return_index=True)
    starts[ep_ids_s, 0] = ep_r[order][first_pos]
    starts[ep_ids_s, 1] = ep_c[order][first_pos]

    loop_ids = np.flatnonzero(starts[:, 0] < 0)
    if loop_ids.size:
        is_loop[loop_ids] = True
        # choose the lexicographically first corner pixel of each loop as artificial vertex
        cr, cc = np.nonzero(chain_id[0::2, 0::2] > 0)
        cid = chain_id[2 * cr, 2 * cc] - 1
        sel = np.isin(cid, loop_ids)
        cr, cc, cid = cr[sel], cc[sel], cid[sel]
        order = np.lexsort((cc, cr, cid))
        cid_s, first_pos = np.unique(cid[order], return_index=True)
        starts[cid_s, 0] = 2 * cr[order][first_pos]
        starts[cid_s, 1] = 2 * cc[order][first_pos]

    coords, offsets = _trace_all(chain_id, starts, sizes)

    # Vertices: all vertex corners plus artificial loop vertices.
    vr, vc = np.nonzero(vertex_corner)
    vertex_cg = np.stack([2 * vr, 2 * vc], axis=1)
    art_cg = starts[is_loop]
    all_cg = np.concatenate([vertex_cg, art_cg], axis=0)
    vertex_kind = np.concatenate(
        [
            np.full(vertex_cg.shape[0], VertexKind.REGULAR, np.int8),
            np.full(art_cg.shape[0], VertexKind.ARTIFICIAL, np.int8),
        ]
    )
    vid_of: Dict[Tuple[int, int], int] = {(int(r), int(c)): k for k, (r, c) in enumerate(all_cg)}

    def cg_to_xy(cg: np.ndarray) -> np.ndarray:
        # corner (i, j) of the padded grid sits at image coords (i - 1.5, j - 1.5)
        return cg.astype(np.float64) / 2.0 - 1.5

    edge_tail = np.empty(n_chains, dtype=np.int64)
    edge_head = np.empty(n_chains, dtype=np.int64)
    edge_faces = np.empty((n_chains, 2), dtype=np.int64)
    edge_polyline: List[np.ndarray] = []

    def adjacent_vertex_corners(r: int, c: int) -> List[Tuple[int, int]]:
        out = []
        if r % 2 == 1:  # vertical crack: corners above/below
            for rr in (r - 1, r + 1):
                if vertex_corner[rr // 2, c // 2]:
                    out.append((rr, c))
        else:  # horizontal crack: corners left/right
            for cc_ in (c - 1, c + 1):
                if vertex_corner[r // 2, cc_ // 2]:
                    out.append((r, cc_))
        return out

    for e in range(n_chains):
        pts = coords[offsets[e] : offsets[e + 1]]
        if is_loop[e]:
            tail_cg = head_cg = (int(pts[0, 0]), int(pts[0, 1]))
            corner_mask = (pts[:, 0] % 2 == 0) & (pts[:, 1] % 2 == 0)
            poly_cg = np.concatenate([pts[corner_mask], pts[:1]], axis=0)
            first_crack = pts[1]
            prev_pt = pts[0]
        else:
            first, last = pts[0], pts[-1]
            t_cands = adjacent_vertex_corners(int(first[0]), int(first[1]))
            h_cands = adjacent_vertex_corners(int(last[0]), int(last[1]))
            if len(pts) == 1:
                tail_cg, head_cg = t_cands[0], t_cands[1]
            else:
                tail_cg, head_cg = t_cands[0], h_cands[0]
            corner_mask = (pts[:, 0] % 2 == 0) & (pts[:, 1] % 2 == 0)
            poly_cg = np.concatenate([np.array([tail_cg]), pts[corner_mask], np.array([head_cg])], axis=0)
            first_crack = pts[0]
            prev_pt = np.array(tail_cg)
        edge_tail[e] = vid_of[tail_cg]
        edge_head[e] = vid_of[head_cg]
        edge_polyline.append(cg_to_xy(poly_cg))

        # Left/right faces from the first crack and its travel direction.
        r, c = int(first_crack[0]), int(first_crack[1])
        dr, dc = r - int(prev_pt[0]), c - int(prev_pt[1])
        if r % 2 == 1:  # vertical crack at column line c/2 between padded pixels (i, c/2-1) and (i, c/2)
            i, j = r // 2, c // 2
            west, east = face_map[i, j - 1], face_map[i, j]
            left, right = (east, west) if dr > 0 else (west, east)
        else:  # horizontal crack at row line r/2 between padded pixels (r/2-1, j) and (r/2, j)
            i, j = r // 2, c // 2
            north, south = face_map[i - 1, j], face_map[i, j]
            left, right = (north, south) if dc > 0 else (south, north)
        edge_faces[e, 0] = left
        edge_faces[e, 1] = right

    cx = HalfEdgeComplex(
        vertex_xy=cg_to_xy(all_cg),
        vertex_kind=vertex_kind,
        edge_tail=edge_tail,
        edge_head=edge_head,
        edge_faces=edge_faces,
        edge_polyline=edge_polyline,
        face_kind=face_kind,
        face_label=face_label,
        face_loops=[[] for _ in range(face_kind.size)],
        he_next=np.zeros(2 * n_chains, dtype=np.int64),
        pixel_size_um=pixel_size_um,
        shape=(H, W),
        provenance={**info, **(provenance or {}), "connectivity": 4},
    )
    rot = build_rotation_system(cx)
    object.__setattr__(cx, "_rotation_cache", rot)
    cx.he_next = next_from_rotation(cx, rot)
    cx.face_loops = _trace_loops(cx)
    return cx


def _trace_loops(cx: HalfEdgeComplex) -> List[List[List[int]]]:
    """Orbits of he_next grouped by the face on their left. Raises if an orbit
    mixes faces, which would indicate a broken rotation convention."""
    n = cx.n_half_edges
    seen = np.zeros(n, dtype=bool)
    he_face = cx.he_face_array()
    loops: List[List[List[int]]] = [[] for _ in range(cx.n_faces)]
    for h0 in range(n):
        if seen[h0]:
            continue
        loop = []
        h = h0
        while not seen[h]:
            seen[h] = True
            loop.append(int(h))
            h = int(cx.he_next[h])
        f = int(he_face[h0])
        if np.any(he_face[loop] != f):
            raise RuntimeError("half-edge loop mixes faces; rotation system inconsistent")
        loops[f].append(loop)
    # Put the loop with the largest absolute area first (outer loop of the face).
    for f in range(cx.n_faces):
        if len(loops[f]) > 1:
            areas = [abs(_loop_signed_area(cx, loop)) for loop in loops[f]]
            order = np.argsort(areas)[::-1]
            loops[f] = [loops[f][k] for k in order]
    return loops


def _loop_signed_area(cx: HalfEdgeComplex, loop: List[int]) -> float:
    pts = []
    for h in loop:
        p = cx.edge_polyline[h >> 1]
        pts.append(p[:-1] if (h & 1) == 0 else p[::-1][:-1])
    P = np.concatenate(pts, axis=0)
    y, x = P[:, 0], P[:, 1]
    # y-up frame so counter-clockwise on screen is positive
    return 0.5 * float(np.sum(x * np.roll(-y, -1) - np.roll(x, -1) * (-y)))
