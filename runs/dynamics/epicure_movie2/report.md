# Dynamics evaluation: epicure_movie2

Label stack: EpiCure curated labels (epics_corrected/abdomen_maxz_z15-24_t1-60_crop_labels.tif, ids track-consistent, no lineage in the label file); 1 px background seams between cells filled with fill_gt_slivers plus a seam pass (100945 px over the movie)

- n_frames: 30
- n_gt_divisions: 0
- pixel_size_um: 0.275
- frame_interval_s: 299.995
- n_ids: 194
- cells_per_frame_min: 104
- cells_per_frame_max: 167

## Event counts (whole sequence)

| kind | gt_ids |
|---|---|
| appearance | 29 |
| contact_birth | 203 |
| contact_death | 122 |
| disappearance | 23 |
| division | 30 |
| extrusion | 94 |
| t1 | 559 |

## Admissibility (exact (dV, dE, dF) identity per frame pair)

| metric | gt_ids |
|---|---|
| n_frame_pairs | 29 |
| frac_pairs_fully_explained | 0.172 |
| frac_pairs_zero_residual | 0.172 |
| n_pairs_without_boundary_events | 9 |
| frac_pairs_fully_explained_no_boundary | 0.556 |
| n_unexplained_events | 52 |
| n_connectivity_changes | 0 |
| residual_abs_sum_histogram | {0: 5, 2: 3, 4: 2, 6: 2, 10: 4, 12: 1, 14: 2, 16: 2, 18: 3, 20: 1, 26: 1, 28: 3} |
| n_extrusion_at_free_edge | 80 |
| n_extrusion_neighbours_meet_at_vertex | 29 |
| n_t1_with_outer_face | 58 |
| n_appearance_at_free_edge | 50 |

## T1 charge conservation

| metric | gt_ids |
|---|---|
| n_t1 | 559 |
| n_t1_charge_conserved | 209 |
| n_t1_generic_side_pattern | 75 |
| n_t1_isolated | 59 |
| n_t1_isolated_charge_conserved | 59 |
| n_t1_isolated_generic_side_pattern | 59 |
| t1_charge_delta_histogram | {-6: 2, -5: 2, -4: 5, -3: 10, -2: 28, -1: 99, 0: 209, 1: 139, 2: 45, 3: 15, 4: 1, 5: 2, 7: 1, 8: 1} |

## Divisions and tracking

### division detection (gt_ids run)

- precision: 0.000
- recall: 0.000
- f1: 0.000
- n_pred: 30
- n_gt: 0
- n_tp: 0
- tolerance_frames: 1

## Largest unexplained residuals (gt_ids)

| frame pair | observed dVEF | expected dVEF | residual | unexplained | diagnosis |
|---|---|---|---|---|---|
| 17 to 18 | (-14, -21, -7) | (-3, -7, -4) | (-11, -14, -3) | {'appearance': 1, 'disappearance': 4} | 4 cell(s) [99, 188, 197, 3299] at the free edge of the annotated region leave the curated ROI (unlabelled tissue beyond, no fixed delta); 1 cell(s) [49] at the free edge of the annotated region enter the curated ROI (unlabelled tissue beyond, no fixed delta); 1 extrusion(s) (1 at the free edge) whose neighbours close the footprint with 1 new contact(s) instead of one vertex: adds about (+1, +1, 0) beyond the table |
| 21 to 22 | (-11, -16, -5) | (-1, -2, -1) | (-10, -14, -4) | {'disappearance': 4} | 4 cell(s) [7, 49, 192, 1197] at the free edge of the annotated region leave the curated ROI (unlabelled tissue beyond, no fixed delta); 1 extrusion(s) (1 at the free edge) whose neighbours close the footprint with 1 new contact(s) instead of one vertex: adds about (+1, +1, 0) beyond the table |
| 22 to 23 | (11, 16, 5) | (1, 2, 1) | (10, 14, 4) | {'appearance': 4} | 4 cell(s) [7, 49, 99, 192] at the free edge of the annotated region enter the curated ROI (unlabelled tissue beyond, no fixed delta); 1 extrusion(s) (1 at the free edge) whose neighbours close the footprint with 2 new contact(s) instead of one vertex: adds about (+2, +2, 0) beyond the table |
| 16 to 17 | (6, 9, 3) | (-4, -4, 0) | (10, 13, 3) | {'appearance': 3} | 3 cell(s) [99, 198, 3299] at the free edge of the annotated region enter the curated ROI (unlabelled tissue beyond, no fixed delta); 2 extrusion(s) (2 at the free edge) whose neighbours close the footprint with 2 new contact(s) instead of one vertex: adds about (+2, +2, 0) beyond the table |
| 27 to 28 | (1, 2, 1) | (-6, -8, -2) | (7, 10, 3) | {'appearance': 3} | 3 cell(s) [153, 188, 193] at the free edge of the annotated region enter the curated ROI (unlabelled tissue beyond, no fixed delta); 3 extrusion(s) (3 at the free edge) whose neighbours close the footprint with 2 new contact(s) instead of one vertex: adds about (+2, +2, 0) beyond the table |
| 0 to 1 | (-9, -13, -4) | (-2, -4, -2) | (-7, -9, -2) | {'disappearance': 2} | 2 cell(s) [124, 5241] at the free edge of the annotated region leave the curated ROI (unlabelled tissue beyond, no fixed delta); 2 extrusion(s) (2 at the free edge) whose neighbours close the footprint with 2 new contact(s) instead of one vertex: adds about (+2, +2, 0) beyond the table |
| 4 to 5 | (0, -1, -1) | (-7, -10, -3) | (7, 9, 2) | {'appearance': 2} | 2 cell(s) [1197, 5241] at the free edge of the annotated region enter the curated ROI (unlabelled tissue beyond, no fixed delta); 2 extrusion(s) (2 at the free edge) whose neighbours close the footprint with 2 new contact(s) instead of one vertex: adds about (+2, +2, 0) beyond the table |
| 6 to 7 | (-4, -6, -2) | (-11, -15, -4) | (7, 9, 2) | {'appearance': 2} | 2 cell(s) [1197, 1557] at the free edge of the annotated region enter the curated ROI (unlabelled tissue beyond, no fixed delta); 3 extrusion(s) (2 at the free edge) whose neighbours close the footprint with 3 new contact(s) instead of one vertex: adds about (+3, +3, 0) beyond the table |
| 12 to 13 | (-6, -9, -3) | (0, -1, -1) | (-6, -8, -2) | {'appearance': 1, 'disappearance': 3} | 3 cell(s) [34, 40, 49] at the free edge of the annotated region leave the curated ROI (unlabelled tissue beyond, no fixed delta); 1 cell(s) [35] at the free edge of the annotated region enter the curated ROI (unlabelled tissue beyond, no fixed delta); 1 extrusion(s) (1 at the free edge) whose neighbours close the footprint with 1 new contact(s) instead of one vertex: adds about (+1, +1, 0) beyond the table |
| 14 to 15 | (-13, -19, -6) | (-7, -11, -4) | (-6, -8, -2) | {'disappearance': 2} | 2 cell(s) [38, 41] at the free edge of the annotated region leave the curated ROI (unlabelled tissue beyond, no fixed delta); 2 extrusion(s) (2 at the free edge) whose neighbours close the footprint with 3 new contact(s) instead of one vertex: adds about (+3, +3, 0) beyond the table |

