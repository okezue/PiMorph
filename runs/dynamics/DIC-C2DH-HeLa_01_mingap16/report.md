# Dynamics evaluation: DIC-C2DH-HeLa_01_mingap16

Label stack: CTC silver-truth masks (xx_ST/SEG) relabelled with the gold TRA marker ids (gold TRA images are markers, not masks); mask/marker stats {'n_masks': 1104, 'n_single': 1095, 'n_split': 9, 'n_no_marker': 0, 'n_marker_without_mask': 16}

- n_frames: 84
- n_gt_divisions: 8
- min_gap_px: 16

## Event counts (whole sequence)

| kind | tracker | gt_ids |
|---|---|---|
| appearance | 8 | 3 |
| contact_birth | 206 | 210 |
| contact_death | 213 | 216 |
| disappearance | 4 | 2 |
| division | 9 | 8 |
| entry | 19 | 13 |
| exit | 21 | 13 |
| extrusion | 3 | 1 |
| gap_appearance | 44 | 44 |
| gap_closure | 3 | 3 |
| gap_disappearance | 44 | 44 |
| gap_nucleation | 1 | 1 |
| reseal | 27 | 27 |
| rupture | 27 | 27 |
| t1 | 3 | 3 |

## Admissibility (exact (dV, dE, dF) identity per frame pair)

| metric | tracker | gt_ids |
|---|---|---|
| n_frame_pairs | 83 | 83 |
| frac_pairs_fully_explained | 0.000 | 0.000 |
| frac_pairs_zero_residual | 0.024 | 0.024 |
| n_pairs_without_boundary_events | 0 | 0 |
| frac_pairs_fully_explained_no_boundary | None | None |
| n_unexplained_events | 474 | 460 |
| n_connectivity_changes | 334 | 341 |
| residual_abs_sum_histogram | {0: 2, 1: 3, 2: 5, 3: 7, 4: 10, 5: 6, 6: 8, 7: 5, 8: 6, 9: 2, 10: 6, 11: 4, 12: 4, 13: 2, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 3, 21: 2, 24: 1, 30: 1, 34: 1} | {0: 2, 1: 3, 2: 4, 3: 7, 4: 11, 5: 6, 6: 8, 7: 5, 8: 6, 9: 2, 10: 5, 11: 3, 12: 4, 13: 4, 14: 2, 15: 1, 16: 1, 17: 1, 18: 1, 19: 2, 21: 2, 24: 1, 30: 1, 34: 1} |

## T1 charge conservation

| metric | tracker | gt_ids |
|---|---|---|
| n_t1 | 3 | 3 |
| n_t1_charge_conserved | 0 | 0 |
| n_t1_generic_side_pattern | 0 | 0 |
| n_t1_isolated | 0 | 0 |
| n_t1_isolated_charge_conserved | 0 | 0 |
| n_t1_isolated_generic_side_pattern | 0 | 0 |
| t1_charge_delta_histogram | {-2: 1, -1: 1, 4: 1} | {-2: 1, -1: 1, 4: 1} |

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

