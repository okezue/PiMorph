# Dynamics evaluation: tissueminer_demo

Label stack: Tissue Analyzer tracked_cells_resized.tif decoded (id = R*65536 + G*256 + B), 1 px white boundary lattice handed to the nearest cell, ids mapped to demo.sqlite cell_id through cell_histories; stats {'n_cell_frames': 44977, 'n_unmapped_cell_frames': 7348}

- n_frames: 71
- n_gt_divisions: 27

## Event counts (whole sequence)

| kind | tracker | gt_ids | database |
|---|---|---|---|
| appearance | 39 | 74 | 1 |
| contact_birth | 495 | 627 | 593 |
| contact_death | 401 | 616 | 509 |
| disappearance | 121 | 102 | 4 |
| division | 192 | 83 | 31 |
| entry | 651 | 462 | 342 |
| exit | 774 | 473 | 408 |
| extrusion | 37 | 94 | 6 |
| t1 | 1617 | 1619 | 1086 |

## Admissibility (exact (dV, dE, dF) identity per frame pair)

| metric | tracker | gt_ids | database |
|---|---|---|---|
| n_frame_pairs | 70 | 70 | 70 |
| frac_pairs_fully_explained | 0.000 | 0.000 | 0.000 |
| frac_pairs_zero_residual | 0.014 | 0.000 | 0.000 |
| n_pairs_without_boundary_events | 0 | 0 | 0 |
| frac_pairs_fully_explained_no_boundary | None | None | None |
| n_unexplained_events | 1585 | 1111 | 755 |
| n_connectivity_changes | 0 | 0 | 0 |
| residual_abs_sum_histogram | {0: 1, 2: 2, 4: 1, 6: 1, 8: 5, 10: 5, 12: 4, 14: 3, 16: 2, 18: 1, 20: 3, 24: 5, 26: 4, 30: 1, 32: 1, 36: 1, 38: 2, 40: 2, 44: 1, 46: 2, 48: 5, 50: 4, 52: 1, 54: 1, 56: 1, 58: 2, 60: 1, 64: 1, 68: 2, 72: 2, 92: 1, 100: 1, 108: 1} | {2: 3, 4: 2, 6: 3, 8: 3, 10: 3, 12: 2, 14: 5, 16: 3, 18: 4, 20: 2, 22: 3, 24: 2, 26: 4, 28: 6, 30: 2, 32: 2, 34: 1, 36: 2, 38: 1, 42: 2, 44: 4, 48: 3, 52: 1, 54: 1, 58: 1, 60: 1, 64: 1, 72: 1, 82: 1, 92: 1} | {2: 3, 4: 5, 6: 3, 8: 1, 10: 8, 12: 5, 16: 2, 18: 4, 20: 2, 22: 3, 24: 6, 26: 3, 28: 1, 30: 5, 32: 2, 36: 2, 38: 1, 44: 3, 46: 2, 48: 1, 50: 2, 52: 2, 54: 1, 56: 1, 60: 1, 90: 1} |

## T1 charge conservation

| metric | tracker | gt_ids | database |
|---|---|---|---|
| n_t1 | 1617 | 1619 | 1086 |
| n_t1_charge_conserved | 947 | 958 | 774 |
| n_t1_generic_side_pattern | 788 | 803 | 675 |
| n_t1_isolated | 740 | 761 | 625 |
| n_t1_isolated_charge_conserved | 740 | 761 | 623 |
| n_t1_isolated_generic_side_pattern | 740 | 761 | 622 |
| t1_charge_delta_histogram | {-11: 1, -10: 3, -9: 8, -8: 11, -7: 12, -6: 10, -5: 25, -4: 30, -3: 31, -2: 49, -1: 166, 0: 947, 1: 183, 2: 43, 3: 12, 4: 19, 5: 26, 6: 8, 7: 9, 8: 10, 9: 7, 10: 2, 11: 4, 13: 1} | {-11: 1, -10: 2, -9: 6, -8: 5, -7: 14, -6: 8, -5: 24, -4: 27, -3: 28, -2: 48, -1: 171, 0: 958, 1: 188, 2: 41, 3: 16, 4: 22, 5: 21, 6: 7, 7: 8, 8: 6, 9: 9, 10: 4, 11: 2, 13: 2, 14: 1} | {-7: 2, -6: 3, -5: 4, -4: 8, -3: 11, -2: 24, -1: 108, 0: 774, 1: 110, 2: 10, 3: 12, 4: 7, 5: 6, 6: 2, 7: 1, 8: 1, 9: 1, 11: 1, 13: 1} |

## Divisions and tracking

### tracking_metrics (tracker vs GT ids)

- n_gt_tracks: 1313
- n_pred_tracks: 1684
- cell_matching_accuracy: 0.716
- frame_match_rate: 1.000
- n_id_switches: 685
- n_fragmentations: 373
- n_unmatched_cell_frames: 0
- n_pred_tracks_spanning_multiple_gt: 303

### division detection (tracker run, events)

- precision: 0.141
- recall: 1.000
- f1: 0.247
- n_pred: 192
- n_gt: 27
- n_tp: 27
- tolerance_frames: 1

### division detection (tracker run, linker lineage only)

- precision: 0.174
- recall: 0.889
- f1: 0.291
- n_pred: 138
- n_gt: 27
- n_tp: 24
- tolerance_frames: 1

### division detection (gt_ids run)

- precision: 0.325
- recall: 1.000
- f1: 0.491
- n_pred: 83
- n_gt: 27
- n_tp: 27
- tolerance_frames: 1

### tracking_metrics (tracker vs Tissue Analyzer codes)

- n_gt_tracks: 1000
- n_pred_tracks: 1684
- cell_matching_accuracy: 0.635
- frame_match_rate: 1.000
- n_id_switches: 727
- n_fragmentations: 365
- n_unmatched_cell_frames: 0
- n_pred_tracks_spanning_multiple_gt: 42

### t1 vs database (tracker, all)

- n_pred: 1617
- n_gt: 1086
- n_tp: 947
- precision: 0.586
- recall: 0.872
- f1: 0.701

### t1 vs database (tracker, ROI cells only)

- n_pred: 1116
- n_gt: 1086
- n_tp: 947
- precision: 0.849
- recall: 0.872
- f1: 0.860

### division detection (tracker, ROI cells only)

- precision: 0.195
- recall: 0.963
- f1: 0.325
- n_pred: 133
- n_gt: 27
- n_tp: 26
- tolerance_frames: 1

### t1 vs database (gt_ids, all)

- n_pred: 1619
- n_gt: 1086
- n_tp: 975
- precision: 0.602
- recall: 0.898
- f1: 0.721

### t1 vs database (gt_ids, ROI cells only)

- n_pred: 1059
- n_gt: 1086
- n_tp: 975
- precision: 0.921
- recall: 0.898
- f1: 0.909

### division detection (gt_ids, ROI cells only)

- precision: 0.643
- recall: 1.000
- f1: 0.783
- n_pred: 42
- n_gt: 27
- n_tp: 27
- tolerance_frames: 1

### division detection (database run)

- precision: 0.871
- recall: 1.000
- f1: 0.931
- n_pred: 31
- n_gt: 27
- n_tp: 27
- tolerance_frames: 1

