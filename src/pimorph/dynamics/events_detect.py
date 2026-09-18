"""Classify topology changes between consecutive complexes into admissible events.

Both complexes must carry track ids as ``face_label`` (see ``tracking.track``).
Each frame is reduced to a ``TopologySnapshot``: cell set, contact set (unordered
pairs of face ids), vertex incidence sets, gap faces with their bounding cells and
the (V, E, F) counts. Face ids are track ids for cells, 0 for the outer face and
negative pseudo-ids for gap faces (matched between the two frames by their
bounding sets). The detected events carry the fixed (dV, dE, dF) of
``pimorph.complex.events``; ``admissibility_check`` compares their sum with the
observed change, which is an exact QC identity: a nonzero residual means a
topology change that no admissible event explains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..complex.geometry import face_area, face_centroid
from ..complex.halfedge import FaceKind, HalfEdgeComplex

Pair = Tuple[int, int]

# Events with a fixed (dV, dE, dF); n is the side count of the removed/created face.
EXPECTED_DELTA_KINDS = (
    "t1",
    "contact_birth",
    "contact_death",
    "division",
    "extrusion",
    "gap_nucleation",
    "gap_closure",
    "rupture",
    "reseal",
    "death_to_gap",
)
# Bookkeeping kinds without a fixed delta (boundary crossings, segmentation noise).
UNEXPLAINED_KINDS = ("exit", "entry", "appearance", "disappearance", "gap_appearance", "gap_disappearance")


def _pair(a: int, b: int) -> Pair:
    return (a, b) if a <= b else (b, a)


@dataclass
class Event:
    kind: str
    participants: Dict[str, Any]
    frame: int  # index of the first frame; the event happens between frame and frame + 1
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def expected_delta(self) -> Optional[Tuple[int, int, int]]:
        return expected_delta_vef(self)

    def as_record(self) -> Dict[str, Any]:
        d = self.expected_delta
        return {
            "frame": int(self.frame),
            "kind": self.kind,
            "participants": _json_safe(self.participants),
            "dV": None if d is None else d[0],
            "dE": None if d is None else d[1],
            "dF": None if d is None else d[2],
            "evidence": _json_safe(self.evidence),
        }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in (sorted(obj) if isinstance(obj, (set, frozenset)) else obj)]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def expected_delta_vef(event: Event) -> Optional[Tuple[int, int, int]]:
    """Fixed (dV, dE, dF) of an event kind (tables in ``pimorph.complex.events``).

    None for bookkeeping kinds and for contact events that merge or split
    skeleton components (their delta depends on the local configuration).
    """
    k = event.kind
    if event.evidence.get("connectivity_change"):
        return None
    if k == "t1" or k == "death_to_gap":
        return (0, 0, 0)
    if k == "contact_birth":
        return (1, 1, 0)
    if k == "contact_death":
        return (-1, -1, 0)
    if k == "division" or k == "rupture":
        return (2, 3, 1)
    if k == "reseal":
        return (-2, -3, -1)
    if k == "extrusion" or k == "gap_closure":
        n = int(event.participants["sides"])
        return (1 - n, -n, -1)
    if k == "gap_nucleation":
        n = int(event.participants["sides"])
        return (n - 1, n, 1)
    return None


# ---------------------------------------------------------------- snapshot
@dataclass
class TopologySnapshot:
    """Combinatorial content of one frame with track ids as face ids."""

    cells: Set[int]
    contacts: Set[Pair]  # unordered pairs of face ids (cells > 0, outer 0, gaps < 0)
    vertices: Set[FrozenSet[int]]  # incident face-id sets of vertices with >= 3 distinct faces
    gaps: Dict[int, FrozenSet[int]]  # gap pseudo-id -> bounding face ids
    sides: Dict[int, int]  # face id -> side count
    counts: Tuple[int, int, int]  # (V, E, F) including the outer face
    gap_centroid: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    area: Dict[int, float] = field(default_factory=dict)  # cell id -> area (optional, disambiguates divisions)
    component: Dict[int, int] = field(default_factory=dict)  # face id -> skeleton component (optional)
    components_at_border: Set[int] = field(default_factory=set)  # components whose loop reaches the outer face
    border: Optional[Set[int]] = None  # cells cut by the image border (None: use the outer-face contact)

    def same_component(self, p: int, q: int) -> Optional[bool]:
        """False when p and q lie in different skeleton components (None if unknown)."""
        if not self.component:
            return None
        if p == 0 or q == 0:
            other = q if p == 0 else p
            c = self.component.get(other)
            return None if c is None else c in self.components_at_border
        cp, cq = self.component.get(p), self.component.get(q)
        return None if cp is None or cq is None else cp == cq

    @property
    def boundary_cells(self) -> Set[int]:
        """Cells in contact with the outer face."""
        return {b for a, b in self.contacts if a == 0}

    @property
    def border_cells(self) -> Set[int]:
        """Cells that can leave or enter the field of view."""
        return self.boundary_cells if self.border is None else self.border

    @property
    def charge(self) -> int:
        return int(sum(6 - self.sides.get(c, 6) for c in self.cells))

    def neighbours(self, f: int) -> Set[int]:
        out = set()
        for a, b in self.contacts:
            if a == f:
                out.add(b)
            elif b == f:
                out.add(a)
        return out

    def neighbour_map(self) -> Dict[int, Set[int]]:
        nb: Dict[int, Set[int]] = {}
        for a, b in self.contacts:
            nb.setdefault(a, set()).add(b)
            nb.setdefault(b, set()).add(a)
        return nb


def snapshot(cx: HalfEdgeComplex) -> TopologySnapshot:
    """Reduce a complex to its topology snapshot (face_label = track id)."""
    fid = np.empty(cx.n_faces, dtype=np.int64)
    gaps: Dict[int, FrozenSet[int]] = {}
    gap_centroid: Dict[int, Tuple[float, float]] = {}
    k = 0
    for f in range(cx.n_faces):
        kind = int(cx.face_kind[f])
        if kind == FaceKind.CELL:
            fid[f] = int(cx.face_label[f])
        elif kind == FaceKind.OUTER:
            fid[f] = 0
        else:
            k += 1
            fid[f] = -k
    cells = {int(x) for x in fid[cx.cell_faces]} if cx.cell_faces.size else set()
    contacts: Set[Pair] = set()
    for e in range(cx.n_edges):
        a, b = int(fid[cx.edge_faces[e, 0]]), int(fid[cx.edge_faces[e, 1]])
        if a != b:
            contacts.add(_pair(a, b))
    sides: Dict[int, int] = {}
    for f in range(cx.n_faces):
        sides[int(fid[f])] = sides.get(int(fid[f]), 0) + cx.face_sides(f)
    vertices: Set[FrozenSet[int]] = set()
    for v in range(cx.n_vertices):
        inc = frozenset(int(fid[f]) for f in cx.vertex_faces(v))
        if len(inc) >= 3:
            vertices.add(inc)
    for g in cx.gap_faces:
        g = int(g)
        bound = frozenset(int(fid[f]) for f in cx.face_neighbors(g, kinds=(FaceKind.CELL, FaceKind.OUTER)))
        gaps[int(fid[g])] = bound
        gap_centroid[int(fid[g])] = tuple(float(x) for x in face_centroid(cx, g, smoothed=False))
    area: Dict[int, float] = {}
    for f in cx.cell_faces:
        f = int(f)
        area[int(fid[f])] = area.get(int(fid[f]), 0.0) + float(face_area(cx, f, smoothed=False))
    component, at_border = _skeleton_components(cx, fid)
    border: Optional[Set[int]] = None
    if cx.shape is not None:
        H, W = cx.shape
        border = set()
        outer = cx.outer_face
        for e in cx.boundary_edges():
            p = cx.edge_polyline[int(e)]
            if (
                np.any(p[:, 0] <= -0.5)
                or np.any(p[:, 0] >= H - 0.5)
                or np.any(p[:, 1] <= -0.5)
                or np.any(p[:, 1] >= W - 0.5)
            ):
                for f in cx.edge_faces[int(e)]:
                    if int(f) != outer and cx.face_kind[int(f)] == FaceKind.CELL:
                        border.add(int(fid[int(f)]))
    return TopologySnapshot(
        cells=cells,
        contacts=contacts,
        vertices=vertices,
        gaps=gaps,
        sides=sides,
        counts=(cx.n_vertices, cx.n_edges, cx.n_faces),
        gap_centroid=gap_centroid,
        area=area,
        component=component,
        components_at_border=at_border,
        border=border,
    )


def _skeleton_components(cx: HalfEdgeComplex, fid: np.ndarray) -> Tuple[Dict[int, int], Set[int]]:
    """Skeleton component of every non-outer face and the components touching the outer face."""
    if cx.n_edges == 0:
        return {}, set()
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = cx.n_vertices
    a = coo_matrix((np.ones(cx.n_edges), (cx.edge_tail, cx.edge_head)), shape=(n, n))
    _, comp = connected_components(a, directed=False)
    outer = cx.outer_face
    component: Dict[int, int] = {}
    at_border: Set[int] = set()
    for e in range(cx.n_edges):
        c = int(comp[cx.edge_tail[e]])
        for side in (0, 1):
            f = int(cx.edge_faces[e, side])
            if f == outer:
                at_border.add(c)
            else:
                component.setdefault(int(fid[f]), c)
    return component, at_border


def _as_snapshot(x: Union[HalfEdgeComplex, TopologySnapshot]) -> TopologySnapshot:
    return x if isinstance(x, TopologySnapshot) else snapshot(x)


def _relabel_snapshot(s: TopologySnapshot, remap: Dict[int, int]) -> TopologySnapshot:
    def m(x: int) -> int:
        return remap.get(x, x)

    return TopologySnapshot(
        cells=set(s.cells),
        contacts={_pair(m(a), m(b)) for a, b in s.contacts},
        vertices={frozenset(m(x) for x in inc) for inc in s.vertices},
        gaps={m(g): frozenset(m(x) for x in bound) for g, bound in s.gaps.items()},
        sides={m(f): n for f, n in s.sides.items()},
        counts=s.counts,
        gap_centroid={m(g): c for g, c in s.gap_centroid.items()},
        area=dict(s.area),
        component={m(f): c for f, c in s.component.items()},
        components_at_border=set(s.components_at_border),
        border=None if s.border is None else set(s.border),
    )


def _area_mismatch(sa: TopologySnapshot, sb: TopologySnapshot, parent: int, children: Sequence[int]) -> Optional[float]:
    """Relative difference between the mother's area and the daughters' total; None without areas."""
    if parent not in sa.area or any(c not in sb.area for c in children):
        return None
    a = float(sa.area[parent])
    if a <= 0:
        return None
    return abs(a - float(sum(sb.area[c] for c in children))) / a


def align_gaps(sa: TopologySnapshot, sb: TopologySnapshot, min_jaccard: float = 0.5) -> TopologySnapshot:
    """Give the gaps of b the pseudo-ids of the gaps of a they continue.

    Gaps are matched by Jaccard similarity of their bounding cell sets (Hungarian,
    ties broken by centroid distance); unmatched gaps of b get fresh negative ids.
    """
    ga = sorted(sa.gaps)
    gb = sorted(sb.gaps)
    remap: Dict[int, int] = {}
    if ga and gb:
        score = np.zeros((len(ga), len(gb)), dtype=np.float64)
        for i, g in enumerate(ga):
            A = sa.gaps[g]
            for j, h in enumerate(gb):
                B = sb.gaps[h]
                jac = len(A & B) / max(len(A | B), 1)
                if jac >= min_jaccard:
                    d = 0.0
                    if g in sa.gap_centroid and h in sb.gap_centroid:
                        d = float(np.hypot(*(np.subtract(sa.gap_centroid[g], sb.gap_centroid[h]))))
                    score[i, j] = jac + 1.0 / (1.0 + d)
        ii, jj = linear_sum_assignment(score, maximize=True)
        for i, j in zip(ii, jj):
            if score[i, j] > 0:
                remap[gb[j]] = ga[i]
    nxt = min(list(sa.gaps) + [0]) - 1
    for h in gb:
        if h not in remap:
            remap[h] = nxt
            nxt -= 1
    # temporary ids avoid collisions while swapping negative ids
    tmp = {h: -1_000_000 - k for k, h in enumerate(gb)}
    inter = _relabel_snapshot(sb, tmp)
    return _relabel_snapshot(inter, {tmp[h]: remap[h] for h in gb})


# ------------------------------------------------------------- detection
def _quadrilateral_ok(lost: Pair, gained: Pair, contacts_a: Set[Pair]) -> bool:
    p, q = lost
    r, s = gained
    if len({p, q, r, s}) != 4:
        return False
    return all(_pair(x, y) in contacts_a for x in (p, q) for y in (r, s))


def match_t1(lost: Iterable[Pair], gained: Iterable[Pair], contacts_a: Set[Pair]) -> List[Tuple[Pair, Pair]]:
    """Pair lost with gained contacts forming the classic T1 quadrilateral.

    Lost (p, q) and gained (r, s) with four distinct faces such that p and q were
    both in contact with r and s before. One-to-one, most constrained first.
    """
    lost = sorted(set(lost))
    gained = sorted(set(gained))
    cand = {L: [G for G in gained if _quadrilateral_ok(L, G, contacts_a)] for L in lost}
    used: Set[Pair] = set()
    out: List[Tuple[Pair, Pair]] = []
    for L in sorted(cand, key=lambda x: (len(cand[x]), x)):
        for G in cand[L]:
            if G not in used:
                used.add(G)
                out.append((L, G))
                break
    return out


def detect_events_from_snapshots(
    sa: TopologySnapshot,
    sb: TopologySnapshot,
    frame: int = 0,
    lineage: Optional[Dict[int, int]] = None,
    heuristic_divisions: bool = True,
) -> List[Event]:
    """Classify the topology change from snapshot a to snapshot b.

    ``lineage`` maps child track -> parent track (from ``tracking.track``). Without
    it, divisions are recognised heuristically: a vanished cell replaced by two new
    cells in mutual contact whose neighbours were the parent's, or a persisting cell
    with a new neighbour whose other neighbours were all the cell's own (one
    daughter keeps the mother's id).
    """
    sb = align_gaps(sa, sb)
    lineage = lineage or {}
    events: List[Event] = []
    nb_a = sa.neighbour_map()
    nb_b = sb.neighbour_map()
    gone = sorted(sa.cells - sb.cells)
    new = sorted(sb.cells - sa.cells)
    new_gaps = sorted(set(sb.gaps) - set(sa.gaps))
    old_gaps = sorted(set(sa.gaps) - set(sb.gaps))
    explained_cells: Set[int] = set()
    consumed_new_gaps: Set[int] = set()
    consumed_old_gaps: Set[int] = set()
    # contacts that the cell/gap events below account for are excluded from contact events
    excluded: Set[int] = set(gone) | set(new) | set(new_gaps) | set(old_gaps)
    boundary_a = sa.border_cells
    boundary_b = sb.border_cells

    # divisions from lineage (parent ends, two children start)
    kids: Dict[int, List[int]] = {}
    for c in new:
        p = lineage.get(c)
        if p is not None and p in gone:
            kids.setdefault(p, []).append(c)
    for p, cs in sorted(kids.items()):
        if len(cs) == 2:
            events.append(Event("division", {"parent": p, "children": sorted(cs)}, frame, {"source": "lineage"}))
            explained_cells.update([p, *cs])

    # heuristic divisions, symmetric convention
    if heuristic_divisions:
        for p in gone:
            if p in explained_cells:
                continue
            cands = []
            for c in new:
                if c in explained_cells:
                    continue
                ext = nb_b.get(c, set()) - set(new)
                if not ext:
                    continue
                score = len(ext & nb_a.get(p, set())) / len(ext)
                if score >= 0.5:
                    cands.append(c)
            pairs = [(c1, c2) for i, c1 in enumerate(cands) for c2 in cands[i + 1 :] if _pair(c1, c2) in sb.contacts]
            pairs = [pr for pr in pairs if (_area_mismatch(sa, sb, p, pr) or 0.0) < 0.5]
            if len(pairs) == 1:
                c1, c2 = pairs[0]
                events.append(
                    Event("division", {"parent": p, "children": sorted([c1, c2])}, frame, {"source": "heuristic"})
                )
                explained_cells.update([p, c1, c2])
        # asymmetric convention: one daughter keeps the mother's id
        for c in new:
            if c in explained_cells or c in boundary_b:
                continue
            ext = nb_b.get(c, set()) - set(new)
            parents = [p for p in ext if p in sa.cells and p not in gone]
            best = None
            for p in parents:
                rest = ext - {p}
                if not rest:
                    continue
                score = len(rest & (nb_a.get(p, set()) | {0})) / len(rest)
                own = nb_b.get(p, set()) - {c} - set(new)
                score_p = len(own & nb_a.get(p, set())) / max(len(own), 1)
                # the mother's old footprint holds both daughters; neighbours handed
                # over to the daughter break remaining ties
                mismatch = _area_mismatch(sa, sb, p, [p, c])
                if mismatch is not None and mismatch >= 0.5:
                    continue
                transferred = len((nb_a.get(p, set()) & rest) - nb_b.get(p, set()))
                key = (-(mismatch if mismatch is not None else 0.0), transferred, score)
                if score >= 0.5 and score_p >= 0.5 and (best is None or key > best[0]):
                    best = (key, p)
            if best is not None:
                p = best[1]
                events.append(
                    Event(
                        "division",
                        {"parent": p, "children": sorted([p, c])},
                        frame,
                        {"source": "heuristic_asymmetric"},
                    )
                )
                explained_cells.add(c)
                excluded.add(p)

    # remaining vanished cells
    for p in gone:
        if p in explained_cells:
            continue
        N = nb_a.get(p, set())
        N_cells = {x for x in N if x > 0}
        if p in boundary_a:
            events.append(Event("exit", {"cell": p, "sides": sa.sides.get(p)}, frame, {"neighbours": sorted(N)}))
            continue
        gap_hit = [g for g in new_gaps if g not in consumed_new_gaps and len(sb.gaps[g] & N_cells) >= 2]
        if gap_hit:
            g = max(gap_hit, key=lambda g: len(sb.gaps[g] & N_cells))
            consumed_new_gaps.add(g)
            events.append(
                Event(
                    "death_to_gap",
                    {"cell": p, "gap": g, "sides": sa.sides.get(p)},
                    frame,
                    {"bounding_matches": sb.gaps[g] == frozenset(N), "neighbours": sorted(N)},
                )
            )
            continue
        if N_cells and N_cells <= sb.cells:
            meet = any(N <= inc for inc in sb.vertices)
            new_among = [c for c in sb.contacts - sa.contacts if c[0] in N and c[1] in N]
            events.append(
                Event(
                    "extrusion",
                    {"cell": p, "sides": int(sa.sides.get(p, len(N))), "neighbours": sorted(N)},
                    frame,
                    {"neighbours_meet_at_vertex": bool(meet), "new_contacts_among_neighbours": len(new_among)},
                )
            )
            continue
        events.append(Event("disappearance", {"cell": p, "sides": sa.sides.get(p)}, frame, {"neighbours": sorted(N)}))

    # remaining new cells
    for c in new:
        if c in explained_cells:
            continue
        kind = "entry" if c in boundary_b else "appearance"
        events.append(
            Event(kind, {"cell": c, "sides": sb.sides.get(c)}, frame, {"neighbours": sorted(nb_b.get(c, set()))})
        )

    # gap events; a rupture opens a whole contact (gap bounded by the two cells
    # and the cap cells at both ends, contact lost) or part of one (gap bounded
    # by the two cells only, contact kept as two components)
    lost_all = sa.contacts - sb.contacts
    gained_all = sb.contacts - sa.contacts
    consumed_contacts: Set[Pair] = set()

    def opened_contact(S: FrozenSet[int], changed: Set[Pair], nb: Dict[int, Set[int]]) -> Optional[Pair]:
        for p, q in sorted(changed):
            if p in S and q in S and p not in excluded and q not in excluded:
                caps = S - {p, q}
                if caps <= (nb.get(p, set()) & nb.get(q, set())):
                    return (p, q)
        return None

    for g in new_gaps:
        if g in consumed_new_gaps:
            continue
        S = sb.gaps[g]
        rupt = opened_contact(S, lost_all, nb_a)
        if len(S) >= 3 and S in sa.vertices:
            events.append(Event("gap_nucleation", {"gap": g, "cells": sorted(S), "sides": len(S)}, frame))
            consumed_new_gaps.add(g)
        elif rupt is not None:
            events.append(Event("rupture", {"gap": g, "cells": list(rupt), "caps": sorted(S - set(rupt))}, frame))
            consumed_new_gaps.add(g)
            consumed_contacts.add(rupt)
        elif len(S) == 2 and tuple(sorted(S)) in sa.contacts:
            events.append(Event("rupture", {"gap": g, "cells": sorted(S), "caps": []}, frame))
            consumed_new_gaps.add(g)
        else:
            events.append(Event("gap_appearance", {"gap": g, "cells": sorted(S)}, frame))
    for g in old_gaps:
        S = sa.gaps[g]
        closed = opened_contact(S, gained_all, nb_b)
        if len(S) >= 3 and any(S <= inc for inc in sb.vertices):
            events.append(Event("gap_closure", {"gap": g, "cells": sorted(S), "sides": len(S)}, frame))
            consumed_old_gaps.add(g)
        elif closed is not None:
            events.append(Event("reseal", {"gap": g, "cells": list(closed), "caps": sorted(S - set(closed))}, frame))
            consumed_old_gaps.add(g)
            consumed_contacts.add(closed)
        elif len(S) == 2 and tuple(sorted(S)) in sb.contacts:
            events.append(Event("reseal", {"gap": g, "cells": sorted(S), "caps": []}, frame))
            consumed_old_gaps.add(g)
        else:
            events.append(Event("gap_disappearance", {"gap": g, "cells": sorted(S)}, frame))

    # contact events among persisting faces
    def keep(c: Pair) -> bool:
        return c[0] not in excluded and c[1] not in excluded and c not in consumed_contacts

    lost = [c for c in lost_all if keep(c)]
    gained = [c for c in gained_all if keep(c)]
    t1_pairs = match_t1(lost, gained, sa.contacts)
    used_lost = {L for L, _ in t1_pairs}
    used_gained = {G for _, G in t1_pairs}
    for L, G in t1_pairs:
        p, q = L
        r, s = G
        dn = {x: int(sb.sides.get(x, 0) - sa.sides.get(x, 0)) for x in (p, q, r, s)}
        events.append(
            Event(
                "t1",
                {"lost": [p, q], "gained": [r, s], "cells": sorted([p, q, r, s])},
                frame,
                {
                    "side_change": dn,
                    "charge_delta": -int(sum(dn.values())),
                    "generic": dn[p] == -1 and dn[q] == -1 and dn[r] == 1 and dn[s] == 1,
                },
            )
        )
    # a contact that joins or separates two skeleton components (sparse cultures)
    # is not the degree-4 vertex rewrite of the table and keeps no fixed delta
    for c in sorted(lost):
        if c not in used_lost:
            split = sb.same_component(c[0], c[1])
            events.append(
                Event("contact_death", {"cells": [c[0], c[1]]}, frame, {"connectivity_change": split is False})
            )
    for c in sorted(gained):
        if c not in used_gained:
            joined = sa.same_component(c[0], c[1])
            events.append(
                Event("contact_birth", {"cells": [c[0], c[1]]}, frame, {"connectivity_change": joined is False})
            )
    # a T1 is isolated when none of its four faces takes part in another change in
    # this frame pair; only then is its local side pattern a clean conservation test
    touched: Dict[int, int] = {}
    for ev in events:
        for x in _event_faces(ev, sa, sb):
            touched[x] = touched.get(x, 0) + 1
    for ev in events:
        if ev.kind == "t1":
            ev.evidence["isolated"] = all(touched.get(x, 0) == 1 for x in ev.participants["cells"])
    return events


def _event_faces(ev: Event, sa: TopologySnapshot, sb: TopologySnapshot) -> Set[int]:
    """Faces whose side count an event can change."""
    p = ev.participants
    if ev.kind == "t1":
        return set(p["cells"])
    if ev.kind in ("contact_birth", "contact_death", "rupture", "reseal"):
        return set(p["cells"]) | set(p.get("caps", []))
    if ev.kind in ("gap_nucleation", "gap_closure", "gap_appearance", "gap_disappearance"):
        return set(p["cells"])
    if ev.kind == "division":
        cells = {p["parent"], *p["children"]}
        out = set(cells)
        for c in cells:
            out |= sa.neighbours(c) | sb.neighbours(c)
        return out
    c = p.get("cell")
    if c is None:
        return set()
    return {c} | sa.neighbours(c) | sb.neighbours(c)


def detect_events(
    cx_a: Union[HalfEdgeComplex, TopologySnapshot],
    cx_b: Union[HalfEdgeComplex, TopologySnapshot],
    frame: int = 0,
    lineage: Optional[Dict[int, int]] = None,
    heuristic_divisions: bool = True,
) -> List[Event]:
    """Detect admissible events between two complexes whose face labels are track ids."""
    return detect_events_from_snapshots(
        _as_snapshot(cx_a), _as_snapshot(cx_b), frame=frame, lineage=lineage, heuristic_divisions=heuristic_divisions
    )


# --------------------------------------------------------------- summaries
def event_summary(events: Sequence[Event]) -> pd.DataFrame:
    """Counts per kind per frame (one row per frame with events, one column per kind)."""
    if not events:
        return pd.DataFrame({"frame": pd.Series(dtype=np.int64)})
    df = pd.DataFrame([{"frame": int(e.frame), "kind": e.kind} for e in events])
    tab = df.groupby(["frame", "kind"]).size().unstack(fill_value=0)
    tab = tab.reindex(sorted(tab.columns), axis=1).reset_index()
    tab.columns.name = None
    return tab


def admissibility_check(
    cx_a: Union[HalfEdgeComplex, TopologySnapshot],
    cx_b: Union[HalfEdgeComplex, TopologySnapshot],
    events: Sequence[Event],
) -> Dict[str, Any]:
    """Observed (dV, dE, dF) versus the sum of the detected events' fixed deltas.

    ``residual`` is observed minus expected; ``n_unexplained`` counts events without
    a fixed delta. ``fully_explained`` requires a zero residual and no unexplained
    event. The topological charge sum_f (6 - n_f) over cells is reported for both
    frames, together with the per-T1 local charge changes (zero for generic T1s).
    """
    sa, sb = _as_snapshot(cx_a), _as_snapshot(cx_b)
    observed = tuple(int(b - a) for a, b in zip(sa.counts, sb.counts))
    expected = np.zeros(3, dtype=np.int64)
    n_unexplained = 0
    n_explained = 0
    for e in events:
        d = e.expected_delta
        if d is None:
            n_unexplained += 1
        else:
            n_explained += 1
            expected += np.asarray(d, dtype=np.int64)
    residual = tuple(int(o - x) for o, x in zip(observed, expected.tolist()))
    t1_charge = [int(e.evidence.get("charge_delta", 0)) for e in events if e.kind == "t1"]
    t1_generic = [bool(e.evidence.get("generic", False)) for e in events if e.kind == "t1"]
    iso = [e for e in events if e.kind == "t1" and e.evidence.get("isolated")]
    return {
        "observed_dVEF": observed,
        "expected_dVEF": tuple(int(x) for x in expected.tolist()),
        "residual_dVEF": residual,
        "euler_residual": int(observed[0] - observed[1] + observed[2]),
        "n_events": len(events),
        "n_explained": n_explained,
        "n_unexplained": n_unexplained,
        "n_connectivity_changes": int(sum(1 for e in events if e.evidence.get("connectivity_change"))),
        "fully_explained": bool(residual == (0, 0, 0) and n_unexplained == 0),
        "charge_a": sa.charge,
        "charge_b": sb.charge,
        "charge_delta": int(sb.charge - sa.charge),
        "n_t1": len(t1_charge),
        "t1_charge_deltas": t1_charge,
        "n_t1_charge_conserved": int(sum(1 for x in t1_charge if x == 0)),
        "n_t1_generic": int(sum(t1_generic)),
        "n_t1_isolated": len(iso),
        "n_t1_isolated_charge_conserved": int(sum(1 for e in iso if e.evidence.get("charge_delta") == 0)),
        "n_t1_isolated_generic": int(sum(1 for e in iso if e.evidence.get("generic"))),
    }


__all__ = [
    "EXPECTED_DELTA_KINDS",
    "Event",
    "TopologySnapshot",
    "UNEXPLAINED_KINDS",
    "admissibility_check",
    "align_gaps",
    "detect_events",
    "detect_events_from_snapshots",
    "event_summary",
    "expected_delta_vef",
    "match_t1",
    "snapshot",
]
