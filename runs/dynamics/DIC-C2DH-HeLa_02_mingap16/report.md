# Dynamics evaluation: DIC-C2DH-HeLa_02_mingap16

Label stack: CTC silver-truth masks (xx_ST/SEG) relabelled with the gold TRA marker ids (gold TRA images are markers, not masks); mask/marker stats {'n_masks': 988, 'n_single': 982, 'n_split': 6, 'n_no_marker': 0, 'n_marker_without_mask': 42}

- n_frames: 84
- n_gt_divisions: 5
- min_gap_px: 16

## Event counts (whole sequence)

| kind | tracker | gt_ids |
|---|---|---|
| appearance | 1 | 1 |
| contact_birth | 204 | 203 |
| contact_death | 218 | 221 |
| disappearance | 0 | 1 |
| division | 7 | 8 |
| entry | 21 | 21 |
| exit | 21 | 22 |
| extrusion | 1 | 0 |
| gap_appearance | 58 | 58 |
| gap_closure | 2 | 2 |
| gap_disappearance | 53 | 53 |
| gap_nucleation | 1 | 1 |
| reseal | 53 | 53 |
| rupture | 49 | 49 |

## Admissibility (exact (dV, dE, dF) identity per frame pair)

| metric | tracker | gt_ids |
|---|---|---|
| n_frame_pairs | 83 | 83 |
| frac_pairs_fully_explained | 0.012 | 0.012 |
| frac_pairs_zero_residual | 0.024 | 0.036 |
| n_pairs_without_boundary_events | 2 | 2 |
| frac_pairs_fully_explained_no_boundary | 0.500 | 0.500 |
| n_unexplained_events | 500 | 504 |
| n_connectivity_changes | 346 | 348 |
| residual_abs_sum_histogram | {0: 2, 1: 3, 2: 4, 3: 6, 4: 4, 5: 6, 6: 5, 7: 7, 8: 8, 9: 3, 10: 5, 11: 1, 12: 6, 13: 3, 14: 2, 15: 1, 16: 1, 17: 2, 18: 1, 19: 1, 20: 2, 23: 1, 25: 1, 26: 1, 28: 1, 30: 1, 32: 1, 35: 1, 38: 1, 42: 1, 47: 1} | {0: 3, 1: 4, 2: 4, 3: 6, 4: 4, 5: 5, 6: 4, 7: 7, 8: 8, 9: 3, 10: 5, 11: 1, 12: 6, 13: 3, 14: 2, 15: 1, 16: 1, 17: 2, 18: 1, 19: 1, 20: 2, 23: 1, 25: 1, 26: 1, 28: 1, 30: 1, 32: 1, 35: 1, 38: 1, 42: 1, 47: 1} |

## T1 charge conservation

| metric | tracker | gt_ids |
|---|---|---|
| n_t1 | 0 | 0 |
| n_t1_charge_conserved | 0 | 0 |
| n_t1_generic_side_pattern | 0 | 0 |
| n_t1_isolated | 0 | 0 |
| n_t1_isolated_charge_conserved | 0 | 0 |
| n_t1_isolated_generic_side_pattern | 0 | 0 |
| t1_charge_delta_histogram | {} | {} |

## Divisions and tracking

### tracking_metrics (tracker vs GT ids)

- n_gt_tracks: 32
- n_pred_tracks: 43
- cell_matching_accuracy: 0.812
- frame_match_rate: 1.000
- n_id_switches: 15
- n_fragmentations: 6
- n_unmatched_cell_frames: 0
- n_pred_tracks_spanning_multiple_gt: 15

### division detection (tracker run, events)

- precision: 0.286
- recall: 0.400
- f1: 0.333
- n_pred: 7
- n_gt: 5
- n_tp: 2
- tolerance_frames: 1

### division detection (tracker run, linker lineage only)

- precision: 0.500
- recall: 0.400
- f1: 0.444
- n_pred: 4
- n_gt: 5
- n_tp: 2
- tolerance_frames: 1

### division detection (gt_ids run)

- precision: 0.625
- recall: 1.000
- f1: 0.769
- n_pred: 8
- n_gt: 5
- n_tp: 5
- tolerance_frames: 1

