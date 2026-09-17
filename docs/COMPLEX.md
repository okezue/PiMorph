# The half-edge cell complex

This document fixes the conventions of `pimorph.complex` so that numbers written by one module can be read by another without guessing. Every statement below corresponds to code in `src/pimorph/complex/` and is exercised by `tests/pimorph/test_complex_extract.py`, `test_events.py`, and `test_matching.py`.

## Coordinates

- Image coordinates are `(row, col)`. Pixel `(r, c)` is centered at `(r, c)` and occupies `[r - 0.5, r + 0.5) x [c - 0.5, c + 0.5)`.
- Crack corners (the points where four pixels meet) therefore sit on half-integers. A vertex at the corner shared by pixels `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)` has coordinates `(0.5, 0.5)`.
- "On screen" means row increases downward and column increases rightward. Angles for the rotation system are measured in a y-up frame (`x = col`, `y = -row`) so that increasing angle is visually counter-clockwise (`halfedge.build_rotation_system`).
- Geometry is stored in pixels. Micrometer values are derived from `pixel_size_um` when it is known; when it is not, tables carry NaN in the `*_um` columns and provenance says `units = "px"` (`io/schema.py`).

## Container: `HalfEdgeComplex`

| Field | Shape | Meaning |
|---|---|---|
| `vertex_xy` | (V, 2) float64 | `(row, col)` of each vertex, crack corner or refined position |
| `vertex_kind` | (V,) int8 | `VertexKind.REGULAR` (degree 3 or 4 at extraction) or `VertexKind.ARTIFICIAL` (degree 2, see below) |
| `edge_tail`, `edge_head` | (E,) int64 | Endpoints of each edge |
| `edge_faces` | (E, 2) int64 | `[face left of the forward half-edge, face right]` |
| `edge_polyline` | list of (n_k, 2) | Exact crack trace from tail to head; first point equals `vertex_xy[tail]`, last equals `vertex_xy[head]` |
| `edge_smooth` | list or None | Subpixel smoothed trace, filled by `geometry.smooth_complex` |
| `face_kind` | (F,) int8 | `FaceKind.CELL = 0`, `GAP = 1`, `OUTER = 2` |
| `face_label` | (F,) int64 | Original label for cell faces, `-1` for gap and outer faces |
| `face_loops` | list of lists of half-edge lists | Oriented boundary loops per face; the first loop is the outer loop, later loops are holes |
| `he_next` | (2E,) int64 | Next half-edge around the face on the left |
| `pixel_size_um`, `shape`, `provenance` | | Calibration, image shape, and a JSON-friendly dictionary of extraction and event history |

## Half-edge ids

Edge `e` has two half-edges: `2*e` runs tail to head (forward) and `2*e + 1` is its twin. `twin(h) = h ^ 1`, `edge_of(h) = h >> 1`, `is_forward(h) = (h & 1) == 0`. `he_face(h) = edge_faces[h >> 1, h & 1]` is the face on the LEFT of `h` when travelling along it on screen. `he_origin(2*e) = edge_tail[e]` and `he_origin(2*e + 1) = edge_head[e]`.

`he_next(h)` is the outgoing half-edge at `head(h)` immediately clockwise from `twin(h)` in the rotation system (`halfedge.next_from_rotation`). Following `he_next` keeps the same face on the left, and the orbits of `he_next` are exactly the face loops. `extract._trace_loops` raises if an orbit mixes faces, which would mean the rotation convention was broken.

## Face kinds

- Cell face: every 4-connected component of a nonzero label. A label made of several components is split into several faces with the same `face_label`; the count is recorded as `provenance["n_split_labels"]`. Cell faces come first in face order (in `skimage.measure.label(connectivity=1)` order).
- Gap face: a 4-connected background component that does not touch the image border. Gap faces follow the cell faces.
- Outer face: exactly one, always last, formed by merging all background components that touch the image border together with the padding rim outside the image. `cx.outer_face` raises if there is not exactly one.

Gap versus outer rule, in one line: background component touches row 0, the last row, column 0, or the last column, then outer; otherwise gap.

## Vertices and artificial vertices

A crack corner is a vertex when 3 or 4 cracks meet at it (`deg >= 3`). Degree 4 arises at pinch corners and checkerboards and is kept as a degree-4 vertex; `test_pinch_splits_label_and_makes_degree4_vertex` and `test_quadrants_topology` cover it.

A closed crack chain that meets no vertex (a cell fully enclosed by one other face, or an island in the outer face) receives one artificial degree-2 vertex at the lexicographically first corner pixel of the loop, flagged `VertexKind.ARTIFICIAL`. The edge is a loop with `edge_tail == edge_head` (`test_island_artificial_vertex`). Smoothing keeps that point fixed. After a rewrite, `rebuild_topology` reassigns kinds by degree (`<= 2` artificial).

## Orientation and sign conventions

- Cell faces and gap faces are traced counter-clockwise on screen; the outer loop of every interior face has positive signed area in the y-up shoelace formula (`geometry.loop_signed_area`). Hole loops are traced with the face on the left as well, which makes them clockwise around the hole, so `face_area` (sum of signed loop areas) subtracts holes automatically. `test_hole_and_island` checks polygon area equals pixel count with a hole present.
- `B1` (V x E): `-1` at the tail vertex, `+1` at the head vertex of each edge. A loop edge has an empty column.
- `B2` (E x F): `+1` when the forward half-edge lies on a loop of face `f`, `-1` when the backward half-edge does. With the outer face included every edge appears in exactly two faces with opposite sign, so each row of `B2` sums to zero and `B1 @ B2` is identically zero.
- Left normal: for a polyline travelled with tangent `(drow, dcol)`, the left unit normal on screen is `(-dcol, drow)`; for the forward half-edge it points into `edge_faces[e, 0]` (`geometry.polyline_normals`, `test_left_face_convention`). `fields/strip.py` uses the same sign: positive lateral offset `r` is the left side.
- Curvature: `polyline_curvature` is positive when turning counter-clockwise on screen.

## Extraction algorithm (`extract.extract_complex`)

Input: a 2-D integer label image `L` of shape `(H, W)`, background 0, no negative labels.

1. Face map. Split nonzero labels into 4-connected components (cell faces). Label 4-connected background components; those touching the border are the outer face, the rest are gap faces. Pad the face map by one pixel of outer face on every side, giving `(H + 2, W + 2)`.
2. Crack grid. Build a boolean grid `CG` of shape `(2 Hp + 1, 2 Wp + 1)` on the padded map. Row `2i + 1`, column `2(j + 1)` is the vertical crack between padded pixels `(i, j)` and `(i, j + 1)`; row `2(i + 1)`, column `2j + 1` is the horizontal crack between `(i, j)` and `(i + 1, j)`. A crack is set where the two face ids differ.
3. Corner degree. Each corner `(i, j)` (grid position `(2i, 2j)`) counts its four neighboring crack cells. Corners with degree `>= 2` are turned on; corners with degree `>= 3` are vertices.
4. Chains. Remove vertex corners from the grid and label the remaining crack pixels with 4-connectivity (`scipy.ndimage.label`). Each component is one edge chain. Chains with an endpoint are open edges; chains without one are loops and get an artificial vertex. `extract_complex` returns early with a single outer face when there are no chains.
5. Tracing. `_trace_all` (numba-compiled when numba is installed, pure Python otherwise) walks every chain from its start pixel, visiting crack cells and corner cells alternately. The edge polyline is the tail corner, the interior corner points (even-even grid positions), and the head corner, converted to image coordinates by `(grid / 2) - 1.5`.
6. Side faces. The left and right faces of an edge come from the pixels flanking its first crack and the travel direction: on a vertical crack travelling down (`drow > 0`) the east pixel is on the left; on a horizontal crack travelling right (`dcol > 0`) the north pixel is on the left.
7. Topology. Build the rotation system from the first segment direction of every half-edge, derive `he_next`, trace the loops, order each face's loops by decreasing absolute area, and raise if a loop mixes faces.

`validate(cx).ok` is true by construction for any input; `test_random_label_images_always_valid` asserts it with Hypothesis over random small label images, and hand-built fixtures cover holes, islands, border-touching cells, double contacts, and pinch corners.

## Why 4-connectivity

Cracks are placed between 4-adjacent pixels, so the connectivity that makes faces simply connected regions is 4-connectivity. Diagonal-only touching pixels of one label are two faces, and a checkerboard corner is a degree-4 vertex. The legacy `endopigraph.interfaces` also counts 4-neighbor contacts, which is what `metrics.structural.legacy_adjacency_f1` reproduces for comparison. `synth.tissue.enforce_4_connected` repairs synthetic labels to the same rule.

## Euler residual, defect law, Weaire identity

These are representation identities, not biological laws (`invariants.py` docstring). They hold for every valid complex and are used as QC checks.

- Euler residual (`incidence.euler_residual`): `V - E + F - (1 + c)`, with the outer face counted in `F` and `c` the number of connected components of the 1-skeleton (vertices and edges). A connected cellulation of the sphere has `V - E + F = 2`; each island adds a component and one to the expected sum. `validate` reports it and requires 0.
- Euler characteristic (`invariants.euler_characteristic`): `V - E + F` with or without the outer face; without it the value is 1 when the skeleton is connected.
- Defect law residual (`invariants.defect_law_residual`): the residual of `sum_f (6 - n_f) = 6 chi + 2 sum_v (d_v - 3) + E_boundary`, where `f` runs over cell and gap faces (outer excluded), `n_f` is the number of edge occurrences around `f` over all of its loops, `chi = V - E + F_interior`, `d_v` is the vertex degree, and `E_boundary` is the number of edges incident to the outer face. The residual is 0 for every valid complex; the tests assert this after extraction and after every event.
- Topological charge (`invariants.topological_charge`): `q_f = 6 - n_f` per cell face, counting cell-cell and cell-gap sides. `t1_charge_delta` sums the change over faces present before and after; `t1_exchange` raises if it is nonzero.
- Weaire sum rule residual (`invariants.weaire_sum_rule_residual`): residual of `sum_n p_n n m_n = <n^2>` over cell faces, where neighbors are counted by physical interface incidence with multiplicity and, with `cells_only=True` (default), `n_f` counts only sides shared with other cells so that the directed neighbor sum closes. The tests require `|residual| < 1e-9`.

## Smoothing and the arclength bias

Crack traces are staircases. A straight boundary at 45 degrees has crack length `sqrt(2)` times its true length, which is the orientation bias of pixel-count contact lengths in the legacy pipeline. `geometry.smooth_polyline(p, iterations=4, lam=0.5)` applies endpoint-fixed Laplacian smoothing: interior points move by `lam * (0.5 * (p[k-1] + p[k+1]) - p[k])` per iteration, endpoints (the vertices) never move, and for a closed polyline the shared endpoint is the artificial vertex and also stays fixed. `smooth_complex(cx)` fills `edge_smooth` and records `{"iterations", "lam"}` in provenance; `cx.edge_geometry(e)` returns the smoothed trace when present.

Measured on the fixtures used by the tests (`test_diagonal_arclength_bias_removed`, `test_disk_perimeter_and_area`) with the default 4 iterations:

| Case | Raw crack trace | Smoothed trace | Test tolerance |
|---|---|---|---|
| Diagonal boundary, 100 x 100 image, true length `99 sqrt(2)` | +41.4 % | +0.05 % | raw > +35 %, smoothed within 3 % |
| Disk of radius 60 px, perimeter `2 pi 60` | +26.3 % | +0.24 % | within 2 % |
| Disk of radius 60 px, area `pi 60^2` (polygon of smoothed trace) | | -0.32 % | within 1 % |

The pixel-count area of a face is exact and separately available through `geometry.face_pixel_area` and the `area_pixels` column of the cells table.

`geometry.refine_vertices(cx, boundary_prob, radius=1)` moves vertices to the boundary-probability-weighted centroid of a `(2 radius + 1)^2` window. Only the smoothed traces follow the moved endpoints; `edge_polyline` stays the exact crack trace and keeps driving the rotation system, so cyclic order cannot change.

## Events (`complex/events.py`)

Each event checks its preconditions (raising `EventPreconditionError`), applies a local combinatorial rewrite on a copy (or in place with `inplace=True`), calls `rebuild_topology`, runs `validate`, and compares the observed `(dV, dE, dF)` with the expected one, raising `EventRewriteError` on any mismatch. The result is an `EventResult` with the new complex, participants, both deltas, and the validation report; the event is appended to `provenance["events"]`. All expected deltas satisfy `dV - dE + dF = 0`.

| Event | Expected (dV, dE, dF) | Preconditions (summary) | Rewrite |
|---|---|---|---|
| `t1_exchange(cx, e, new_length=2.0)` | (0, 0, 0) | `e` is a cell-cell edge, both endpoints trivalent with cell-only wedges, four distinct cells A, B, C, D and five distinct edges | C and D take over edge id `e`, perpendicular to the old edge, length `new_length`, centered on the old midpoint; A and B lose one side, C and D gain one; total charge unchanged |
| `contact_death(cx, e)` | (-1, -1, 0) | Same as T1 | Collapse `e` into one degree-4 vertex at its midpoint; tail id survives, head and edge removed, ids compacted |
| `contact_birth(cx, v, new_length=2.0, start=0)` | (1, 1, 0) | `v` has degree 4 with four distinct cell wedges | Split into two trivalent vertices joined by a new edge; the cells in wedges h1-h2 and h3-h0 gain the contact |
| `divide(cx, f, (e1, t1), (e2, t2))` | (2, 3, 1) | `f` is a cell, `e1 != e2` on its outer loop with `f` on exactly one side, `0 < t < 1` | Chord between the two split points; the arc from p to q goes to a new face with the same `face_label`; `provenance["lineage"]` records child and parent |
| `extrude(cx, f)` | (1 - n, -n, -1) | `f` is an n-sided cell (`n >= 3`) with one loop, distinct interior edges and vertices, trivalent corners, not touching the outer face | All boundary vertices merge into one vertex at the centroid; every neighbor re-closes through it (T2) |
| `nucleate_gap(cx, v, radius=1.0)` | (2, 3, 1) | `v` trivalent with three distinct cell wedges; incident edges longer than `radius` | New vertices at arclength `radius` along each incident edge; three new edges close a triangular GAP face with label -1 |
| `rupture(cx, e, width=1.0)` | (2, 3, 1) | `e` cell-cell with trivalent endpoints not touching the outer face | Two offset copies of the trace (`+width/2` toward A keeps id `e`, `-width/2` toward B is new) plus two straight caps form a 4-sided GAP face |
| `reseal(cx, g)` | (-2, -3, -1) | `g` is a 4-sided GAP with distinct trivalent corners, long sides bounded by two distinct cells | Collapse to one cell-cell edge along the mean of the two long traces; ids compacted |

`test_rupture_then_reseal_roundtrip` and `test_contact_birth_then_death_roundtrip` check that the inverse pairs restore the counts and side numbers. Geometry after a rewrite is combinatorially consistent but not guaranteed free of self-intersections; smoothed traces are dropped by every event and must be recomputed with `smooth_complex`.

## Matching two complexes (`complex/matching.py`)

Both complexes must come from label images of the same shape.

- Faces (`match_faces`): pixel IoU between cell faces, maximum-IoU assignment (Hungarian) restricted to overlapping pairs, kept when IoU `>= iou_thresh` (default 0.5). A reference face overlapped by two or more predicted faces each covering at least `split_frac = 0.2` of its area is a split; the symmetric case is a merge.
- Edges (`match_edges`): for each predicted cell-cell edge whose two faces are matched, candidate reference edges are those between the mapped face pair (`edges_between`). A candidate is accepted when at least `min_overlap = 0.5` of the points of each trace lie within `tol_px = 3.0` of the other trace; ties are resolved greedily by descending overlap, one to one.
  - Pair-level adjacency precision, recall, and F1 count unordered face pairs: predicted pairs mapped through the face matching versus reference pairs. Two disconnected contacts between the same two cells are one pair.
  - Component-level precision, recall, and F1 count matched edges over `n_pred_edges` and `n_ref_edges`, so multiplicity is scored (`test_component_level_sees_double_contact`).
  - Arclength relative error is reported on matched pairs from the smoothed traces.
- Vertices (`match_vertices`): junction vertices (at least three incident cells) are matched by mutual nearest neighbor within `dist_px = 3.0`. For each matched pair, the predicted vertex's cyclic list of incident cells is mapped through the face matching. Incident-set accuracy is the fraction of matched pairs whose mapped set equals the reference set; cyclic-order accuracy additionally requires the mapped cycle to be a rotation of the reference cycle in the same direction (`_cyclic_equal`), so it implies set agreement.
- Complex edit distance (approximate), in `metrics.structural.structural_metrics`: `unmatched predicted faces + unmatched reference faces + unmatched predicted edges + unmatched reference edges + incident-set mismatches`. It is a lower-bound-style count of local repairs, not a true edit distance.

## Dual graph views (`complex/dual.py`)

`to_multigraph` keeps one edge per connected contact component; `to_simple_graph` collapses multiplicity into `n_components` and summed `arclength_px`. Both lose cyclic order, gap faces, and geometry. `clique_vertex_report` counts dual-graph 3-cliques and how many are realized by a vertex incident to all three cells (a degree-4 vertex with four cells realizes four triples).

## CLI

```bash
pimorph validate --labels path/to/labels.tif [--pixel-size-um 0.325]
```

Prints the `ValidationReport` as JSON plus `defect_law_residual` and `weaire_residual`, and exits 1 when the report is not ok.
