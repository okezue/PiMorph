# Dynamics evaluation: DIC-C2DH-HeLa_01

Label stack: CTC silver-truth masks (xx_ST/SEG) relabelled with the gold TRA marker ids (gold TRA images are markers, not masks); mask/marker stats {'n_masks': 1104, 'n_single': 1095, 'n_split': 9, 'n_no_marker': 0, 'n_marker_without_mask': 16}

- n_frames: 84
- n_gt_divisions: 8

## Event counts (whole sequence)

| kind | tracker | gt_ids |
|---|---|---|
| appearance | 8 | 3 |
| contact_birth | 207 | 211 |
| contact_death | 216 | 219 |
| disappearance | 4 | 2 |
| division | 9 | 8 |
| entry | 19 | 13 |
| exit | 21 | 13 |
| extrusion | 3 | 1 |
| gap_appearance | 374 | 378 |
| gap_closure | 3 | 3 |
| gap_disappearance | 469 | 468 |
| gap_nucleation | 2 | 2 |
| reseal | 267 | 268 |
| rupture | 367 | 363 |
| t1 | 4 | 4 |

## Admissibility (exact (dV, dE, dF) identity per frame pair)

| metric | tracker | gt_ids |
|---|---|---|
| n_frame_pairs | 83 | 83 |
| frac_pairs_fully_explained | 0.000 | 0.000 |
| frac_pairs_zero_residual | 0.012 | 0.012 |
| n_pairs_without_boundary_events | 0 | 0 |
| frac_pairs_fully_explained_no_boundary | None | None |
| n_unexplained_events | 1220 | 1209 |
| n_connectivity_changes | 325 | 332 |
| residual_abs_sum_histogram | {0: 1, 1: 1, 3: 3, 4: 2, 5: 2, 6: 2, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 3, 15: 1, 17: 2, 18: 5, 19: 1, 20: 1, 21: 1, 22: 4, 23: 1, 24: 1, 25: 1, 26: 1, 28: 1, 31: 1, 32: 2, 34: 2, 35: 1, 36: 1, 37: 1, 39: 2, 40: 2, 42: 1, 43: 1, 44: 2, 46: 2, 47: 1, 53: 1, 54: 1, 55: 1, 58: 1, 59: 1, 61: 1, 62: 1, 67: 1, 68: 1, 71: 1, 88: 1, 105: 1, 113: 1, 118: 1, 129: 1, 154: 1, 308: 1} | {0: 1, 1: 1, 3: 3, 4: 2, 5: 2, 6: 2, 7: 1, 8: 2, 9: 1, 10: 2, 11: 2, 12: 2, 13: 2, 14: 3, 15: 2, 16: 1, 17: 2, 18: 5, 19: 1, 20: 1, 21: 1, 22: 3, 23: 1, 24: 1, 25: 1, 26: 2, 28: 1, 31: 1, 32: 2, 34: 1, 35: 1, 36: 1, 37: 1, 39: 2, 40: 2, 42: 1, 43: 1, 44: 2, 46: 2, 47: 1, 49: 1, 53: 1, 54: 1, 58: 1, 59: 1, 61: 1, 62: 1, 67: 1, 68: 1, 81: 1, 88: 1, 105: 1, 113: 1, 118: 1, 129: 1, 154: 1, 308: 1} |

## T1 charge conservation

| metric | tracker | gt_ids |
|---|---|---|
| n_t1 | 4 | 4 |
| n_t1_charge_conserved | 0 | 0 |
| n_t1_generic_side_pattern | 0 | 0 |
| n_t1_isolated | 0 | 0 |
| n_t1_isolated_charge_conserved | 0 | 0 |
| n_t1_isolated_generic_side_pattern | 0 | 0 |
| t1_charge_delta_histogram | {-36: 1, -8: 2, -1: 1} | {-36: 1, -8: 2, -1: 1} |

## Divisions and tracking

### tracking_metrics (tracker vs GT ids)

- n_gt_tracks: 37
- n_pred_tracks: 53
- cell_matching_accuracy: 0.784
- frame_match_rate: 1.000
- n_id_switches: 18
- n_fragmentations: 8
- n_unmatched_cell_frames: 0
- n_pred_tracks_spanning_multiple_gt: 12

### division detection (tracker run, events)

- precision: 0.778
- recall: 0.875
- f1: 0.824
- n_pred: 9
- n_gt: 8
- n_tp: 7
- tolerance_frames: 1

### division detection (tracker run, linker lineage only)

- precision: 1.000
- recall: 0.875
- f1: 0.933
- n_pred: 7
- n_gt: 8
- n_tp: 7
- tolerance_frames: 1

### division detection (gt_ids run)

- precision: 1.000
- recall: 1.000
- f1: 1.000
- n_pred: 8
- n_gt: 8
- n_tp: 8
- tolerance_frames: 1

