# Dynamics evaluation: DIC-C2DH-HeLa_02

Label stack: CTC silver-truth masks (xx_ST/SEG) relabelled with the gold TRA marker ids (gold TRA images are markers, not masks); mask/marker stats {'n_masks': 988, 'n_single': 982, 'n_split': 6, 'n_no_marker': 0, 'n_marker_without_mask': 42}

- n_frames: 84
- n_gt_divisions: 5

## Event counts (whole sequence)

| kind | tracker | gt_ids |
|---|---|---|
| appearance | 2 | 3 |
| contact_birth | 222 | 219 |
| contact_death | 242 | 244 |
| disappearance | 0 | 1 |
| division | 6 | 7 |
| entry | 21 | 21 |
| exit | 21 | 22 |
| extrusion | 1 | 1 |
| gap_appearance | 559 | 559 |
| gap_closure | 3 | 3 |
| gap_disappearance | 535 | 535 |
| gap_nucleation | 1 | 1 |
| reseal | 702 | 702 |
| rupture | 688 | 688 |
| t1 | 1 | 1 |

## Admissibility (exact (dV, dE, dF) identity per frame pair)

| metric | tracker | gt_ids |
|---|---|---|
| n_frame_pairs | 83 | 83 |
| frac_pairs_fully_explained | 0.012 | 0.012 |
| frac_pairs_zero_residual | 0.012 | 0.012 |
| n_pairs_without_boundary_events | 2 | 2 |
| frac_pairs_fully_explained_no_boundary | 0.500 | 0.500 |
| n_unexplained_events | 1511 | 1513 |
| n_connectivity_changes | 373 | 372 |
| residual_abs_sum_histogram | {0: 1, 4: 2, 6: 1, 7: 2, 8: 2, 9: 2, 10: 1, 11: 2, 12: 5, 14: 2, 15: 2, 16: 2, 18: 2, 19: 1, 20: 2, 21: 2, 22: 2, 23: 1, 24: 1, 25: 3, 27: 1, 28: 1, 29: 1, 30: 1, 32: 1, 33: 1, 34: 2, 35: 2, 39: 2, 40: 2, 45: 1, 49: 4, 57: 1, 58: 1, 66: 1, 70: 1, 74: 1, 78: 1, 83: 1, 85: 1, 89: 1, 93: 1, 94: 1, 97: 1, 101: 1, 106: 1, 113: 1, 121: 1, 124: 1, 132: 1, 137: 1, 138: 1, 141: 1, 150: 1, 159: 1, 207: 1, 246: 1, 415: 1} | {0: 1, 4: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 1, 11: 2, 12: 4, 14: 2, 15: 3, 16: 2, 18: 2, 19: 1, 20: 2, 21: 1, 22: 2, 23: 1, 24: 1, 25: 3, 27: 1, 28: 1, 29: 1, 30: 1, 32: 1, 33: 1, 34: 2, 35: 2, 39: 2, 40: 2, 45: 1, 49: 4, 57: 1, 58: 1, 66: 1, 70: 1, 74: 1, 78: 1, 83: 1, 85: 1, 89: 1, 91: 1, 94: 1, 101: 1, 103: 1, 106: 1, 113: 1, 121: 1, 124: 1, 132: 1, 137: 1, 138: 1, 141: 1, 150: 1, 159: 1, 207: 1, 246: 1, 415: 1} |

## T1 charge conservation

| metric | tracker | gt_ids |
|---|---|---|
| n_t1 | 1 | 1 |
| n_t1_charge_conserved | 0 | 0 |
| n_t1_generic_side_pattern | 0 | 0 |
| n_t1_isolated | 0 | 0 |
| n_t1_isolated_charge_conserved | 0 | 0 |
| n_t1_isolated_generic_side_pattern | 0 | 0 |
| t1_charge_delta_histogram | {-12: 1} | {-12: 1} |

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

- precision: 0.333
- recall: 0.400
- f1: 0.364
- n_pred: 6
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

- precision: 0.714
- recall: 1.000
- f1: 0.833
- n_pred: 7
- n_gt: 5
- n_tp: 5
- tolerance_frames: 1

