# Dynamics evaluation: epicure_movie3

Label stack: EpiCure curated labels (epics_corrected/moji_merged_3to13_crop_labels.tif, ids track-consistent, no lineage in the label file); 1 px background seams between cells filled with fill_gt_slivers plus a seam pass (116444 px over the movie)

- n_frames: 11
- n_gt_divisions: 0
- pixel_size_um: None
- frame_interval_s: None
- n_ids: 239
- cells_per_frame_min: 173
- cells_per_frame_max: 196

## Event counts (whole sequence)

| kind | gt_ids |
|---|---|
| appearance | 2 |
| contact_birth | 12 |
| contact_death | 5 |
| disappearance | 2 |
| division | 33 |
| exit | 1 |
| extrusion | 9 |
| gap_appearance | 1 |
| gap_closure | 10 |
| gap_disappearance | 1 |
| gap_nucleation | 15 |
| reseal | 10 |
| rupture | 11 |
| t1 | 24 |

## Admissibility (exact (dV, dE, dF) identity per frame pair)

| metric | gt_ids |
|---|---|
| n_frame_pairs | 10 |
| frac_pairs_fully_explained | 0.100 |
| frac_pairs_zero_residual | 0.100 |
| n_pairs_without_boundary_events | 6 |
| frac_pairs_fully_explained_no_boundary | 0.167 |
| n_unexplained_events | 7 |
| n_connectivity_changes | 0 |
| residual_abs_sum_histogram | {0: 1, 2: 5, 4: 2, 6: 1, 10: 1} |
| n_extrusion_at_free_edge | 2 |
| n_extrusion_neighbours_meet_at_vertex | 6 |
| n_t1_with_outer_face | 5 |
| n_appearance_at_free_edge | 0 |

## T1 charge conservation

| metric | gt_ids |
|---|---|
| n_t1 | 24 |
| n_t1_charge_conserved | 15 |
| n_t1_generic_side_pattern | 15 |
| n_t1_isolated | 13 |
| n_t1_isolated_charge_conserved | 13 |
| n_t1_isolated_generic_side_pattern | 13 |
| t1_charge_delta_histogram | {-3: 2, -1: 5, 0: 15, 2: 1, 3: 1} |

## Divisions and tracking

### division detection (gt_ids run)

- precision: 0.000
- recall: 0.000
- f1: 0.000
- n_pred: 33
- n_gt: 0
- n_tp: 0
- tolerance_frames: 1

## Largest unexplained residuals (gt_ids)

| frame pair | observed dVEF | expected dVEF | residual | unexplained | diagnosis |
|---|---|---|---|---|---|
| 2 to 3 | (7, 10, 3) | (10, 15, 5) | (-3, -5, -2) | {'disappearance': 1, 'exit': 1} | 1 cell(s) cross the image border (exit/entry, no fixed delta); 1 cell(s) [9] leave inside the tissue: id change or dropped label |
| 5 to 6 | (-9, -13, -4) | (-10, -15, -5) | (1, 2, 1) | {'appearance': 2, 'disappearance': 1} | 1 cell(s) [95] leave inside the tissue: id change or dropped label; 2 cell(s) [14, 1749] enter inside the tissue: id change or dropped label |
| 9 to 10 | (12, 18, 6) | (14, 21, 7) | (-2, -3, -1) | {'gap_disappearance': 1} | 1 background pocket(s) open/close without a matching nucleation/closure/rupture |
| 8 to 9 | (7, 10, 3) | (6, 8, 2) | (1, 2, 1) | {'gap_appearance': 1} | 1 extrusion(s) (0 at the free edge) whose neighbours close the footprint with 1 new contact(s) instead of one vertex: adds about (+1, +1, 0) beyond the table; 1 background pocket(s) open/close without a matching nucleation/closure/rupture |
| 1 to 2 | (-4, -6, -2) | (-3, -5, -2) | (-1, -1, 0) | {} | 1 extrusion(s) (0 at the free edge) whose neighbours close the footprint with 1 new contact(s) instead of one vertex: adds about (+1, +1, 0) beyond the table |
| 3 to 4 | (8, 12, 4) | (9, 13, 4) | (-1, -1, 0) | {} | all events fixed-delta yet residual (-1, -1, 0): vertex set changed by +31/-23 without a matched contact change (fourfold vertex or division geometry) |
| 4 to 5 | (18, 27, 9) | (17, 26, 9) | (1, 1, 0) | {} | all events fixed-delta yet residual (1, 1, 0): vertex set changed by +52/-36 without a matched contact change (fourfold vertex or division geometry) |
| 6 to 7 | (7, 10, 3) | (8, 11, 3) | (-1, -1, 0) | {} | 1 extrusion(s) (0 at the free edge) whose neighbours close the footprint with 1 new contact(s) instead of one vertex: adds about (+1, +1, 0) beyond the table |
| 7 to 8 | (1, 2, 1) | (2, 3, 1) | (-1, -1, 0) | {} | 2 T1(s) with a non-generic side pattern (fourfold vertex or overlapping rewrites), 1 division(s) |

