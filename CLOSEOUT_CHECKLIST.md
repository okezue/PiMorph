# EndoPiGraph-AJmorph Close-Out Checklist

## 1. One-Command Reproduction

### Full Pipeline (single image)
```bash
python3 src/EndoPiGraph_AJmorph_v1_allinone.py \
    --image path/to/image.tif \
    --output runs/my_run \
    --ve-cadherin-channel 0
```

### Batch Processing
```bash
./scripts/run_egm2_batch.sh
# Or manually:
python3 scripts/batch_process.py data/egm2_images/ runs/egm2_batch/
```

### Run B3 Experiment (Typed vs Untyped Classification)
```bash
python3 scripts/typed_vs_untyped_experiment.py runs/egm2_full/
```

### Harden Network Statistics
```bash
python3 scripts/harden_network_stats.py runs/egm2_full/
```

---

## 2. Results Index

| Result | Path | Description |
|--------|------|-------------|
| **QC Results** | `runs/sbiad1540_full/qc_results.csv` | Image quality metrics |
| **Polarity Summary** | `runs/sbiad1540_full/polarity_summary_by_condition.csv` | R/V by condition (B1) |
| **All Cells** | `runs/egm2_full/all_cells.csv` | Aggregated cell morphometrics |
| **All Edges** | `runs/egm2_full/all_edges.csv` | Aggregated edge/junction features |
| **B3 Results** | `runs/egm2_full/typed_vs_untyped_results.json` | Typed vs Untyped comparison |
| **Network Stats** | `runs/egm2_full/hardened_network_stats.json` | Per-image network findings (3-clique keys: `all_reticular_3clique_pct`, `n_3cliques`, `enrichment_z`, `perm_p`) |
| **Classifier Model** | `models/ajmorph_classifier_v2.joblib` | Trained AJ morphology classifier |
| **Classifier Eval** | `models/ajmorph_evaluation_report.json` | Cross-validation results |
| **Network Discoveries** | `NETWORK_DISCOVERIES.md` | Network-level findings (measured vs hypothesized) |
| **Science Summary** | `B1_B3_SCIENCE_TASKS_SUMMARY.md` | B1-B3 task results |

### QC Images (per processed image)
```
runs/egm2_full/{image_id}/
├── cells.csv           # Cell-level features
├── edges.csv           # Edge-level features
├── polarity.csv        # Polarity metrics
├── qc_composite.png    # Visual QC overlay
├── qc_cellpose_seg.png # Segmentation mask
└── qc_skeleton.png     # Junction skeleton
```

---

## 3. Model Evaluation Protocol

### AJ Morphology Classifier

**Method:** 5-fold GroupKFold by `image_id`

- **No leakage:** Patches from the same image never appear in both train and test
- **Evaluation script:** `scripts/train_classifier_proper.py`
- **Results:** `models/ajmorph_evaluation_report.json`

```
Accuracy: 99.6% (5-fold CV)
Macro F1: 0.991

Per-fold accuracies:
  Fold 1: 99.7% (test images: BSA_BMP9_20dyn-8, BSA_static-115, EGM2_regular_static-50)
  Fold 2: 99.9% (test images: BSA_BMP9_6dyn-29, EGM2_regular_18-20dyn-53, EGM2_regular_6dyn-27)
  Fold 3: 99.4% (test images: BSA_BMP9_static-23, BSA_static-111, EGM2_regular_6dyn-88)
  Fold 4: 99.3% (test images: BSA_6dyn-146, EGM2_regular_18-20dyn-06, EGM2_regular_18-20dyn-60)
  Fold 5: 99.7% (test images: BSA_BMP9_20dyn-2, EGM2_regular_6dyn-86, EGM2_regular_static-49)
```

**CRITICAL WARNING:** Labels are heuristic-derived from threshold rules on the same features used for training. High accuracy indicates the RF learned the rules, NOT validated biological classification. Manual annotation is required for true validation.

### B3: Typed vs Untyped Classification

**Method:** Leave-One-Out Cross-Validation (LOOCV) by image

- **Evaluation script:** `scripts/typed_vs_untyped_experiment.py`
- **Results:** `runs/egm2_full/typed_vs_untyped_results.json`

```
Typed features (LR):   F1 = 91.9%
Untyped features (LR): F1 = 80.6%

Conclusion: Junction typing (reticular/punctate/straight/fingers)
provides significant predictive value for flow condition classification.
```

### Network Discovery Statistics

**Method:** Per-image replicate testing (n = images, not cells)

- **Evaluation script:** `scripts/harden_network_stats.py`
- **Results:** `runs/egm2_full/hardened_network_stats.json`

All statistics use Mann-Whitney U (between conditions) or Wilcoxon signed-rank (within condition) with bootstrap 95% CIs.

The all-reticular 3-clique fraction is additionally tested against a conditional null that preserves each image's reticular-edge count (label permutation, 1000 permutations, seed 0), reported as per-image `enrichment_z` and `perm_p`, with conditions compared by Mann-Whitney U on `enrichment_z` (`tests_enrichment_z` in the JSON). A graph 3-clique is a tricellular junction only when the three cells share a physical vertex; `--complex-dir` reports that fraction from the `pimorph` complex when available.

---

## 4. Known Limitations

### Data/Pipeline Limitations

1. **V (mean vector magnitude) depends on flow-axis calibration**
   - R (alignment strength) is axis-independent and reliable
   - V requires knowing the true flow direction from experimental metadata
   - Use R as primary polarity metric unless flow axis is confirmed

2. **QC failures bias toward static condition**
   - Low-confluence images more common in static wells
   - ~15% of images fail QC thresholds

3. **Cellpose segmentation artifacts**
   - Over-segmentation of elongated cells under high flow
   - Under-segmentation at cell junctions
   - Manual verification recommended for edge cases

### Statistical Limitations

4. **Degree-occupancy correlation withdrawn**
   - Original pooled analysis showed p < 1e-17
   - Per-image testing: p = 0.808 (not significant)
   - Demonstrates importance of proper replicate-level testing

5. **AJ morphology labels are heuristic**
   - RF classifier has 99.6% accuracy on heuristic labels
   - Does NOT validate biological classification accuracy
   - Manual annotation needed for true validation

6. **Sample sizes per condition**
   - EGM2: 30 static, 30 6dyne, 30 20dyne (adequate)
   - SBIAD1540: 5-6 per condition (underpowered for some tests)

7. **3-cliques are not tricellular junctions by construction**
   - The all-reticular 3-clique fraction is a graph quantity (three pairwise-adjacent cells)
   - Its raw increase can follow from the reticular-fraction increase alone; the conditional null in `scripts/harden_network_stats.py` tests for enrichment beyond that
   - Whether a 3-clique is realized by a common vertex is checked with `--complex-dir` (pimorph complex); results pending

8. **Geometry channel equals junction channel**
   - Cells were segmented from VE-cadherin, the same channel scored for AJ morphology
   - Recorded as `geometry_source: junction_channel` in `run_summary.json` provenance; the pipeline warns on this configuration
   - Biological interpretations (maturation, adhesion strength, barrier function) are hypotheses, not validated findings

### Naming/Metadata Issues

9. **high_shear naming inconsistency**
   - Some datasets use "high_shear" vs "20dyne"
   - Condition parsing logic handles both but may need review

---

## 5. Repository Structure

```
EndoPiGraph-AJmorph/
├── src/
│   └── EndoPiGraph_AJmorph_v1_allinone.py  # Main pipeline
├── scripts/
│   ├── train_classifier_proper.py          # Classifier training
│   ├── typed_vs_untyped_experiment.py      # B3 experiment
│   ├── harden_network_stats.py             # Network validation
│   └── run_egm2_batch.sh                   # Batch processing
├── models/
│   ├── ajmorph_classifier_v2.joblib        # Trained model
│   └── ajmorph_evaluation_report.json      # Evaluation metrics
├── runs/
│   ├── egm2_full/                          # 102 EGM2 images
│   └── sbiad1540_full/                     # 15 SBIAD1540 images
├── data/                                   # Raw image data
├── NETWORK_DISCOVERIES.md                  # Network-level findings
├── B1_B3_SCIENCE_TASKS_SUMMARY.md          # Science task results
├── CLOSEOUT_CHECKLIST.md                   # This document
└── README.md                               # Project overview
```

---

## 6. Validation Summary

| Component | Validation Method | Status |
|-----------|------------------|--------|
| Patch classifier | GroupKFold by image_id | ✅ No leakage |
| B3 experiment | LOOCV by image | ✅ Typed > Untyped |
| Discovery 1 (clustering) | Per-image Mann-Whitney + density regression | ⚠️ Confounded by mean degree (R² = 0.94); withdrawn |
| Discovery 2 (reticular %) | Per-image Mann-Whitney | ✅ p = 3.0e-06 |
| Discovery 3 (degree-occupancy) | Per-image Wilcoxon | ❌ Withdrawn (p = 0.81) |
| Discovery 4 (all-reticular 3-cliques, raw %) | Per-image Mann-Whitney | ✅ p = 1.5e-04 |
| Discovery 4 (3-clique enrichment_z vs conditional null) | Per-image Mann-Whitney on `enrichment_z` | ⏳ Pending re-run |
| Discovery 4 (3-cliques realized by a common vertex) | `--complex-dir` (pimorph complex) | ⏳ Pending |
| Discovery 5 (area-degree) | Per-image Mann-Whitney | ✅ p = 2.6e-07 |

Statistical status refers to the measured quantities only. Biological interpretations (junction maturation, adhesion strength, barrier function, tricellular hotspots) are hypotheses pending functional validation.

---

## 7. Reproduction Checklist

- [ ] Clone repository: `git clone https://github.com/okezue/EndoPiGraph-AJmorph.git`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Pull LFS files: `git lfs pull`
- [ ] Run single image test: `python3 src/EndoPiGraph_AJmorph_v1_allinone.py --image data/test.tif --output runs/test`
- [ ] Verify QC outputs in `runs/test/`
- [ ] Run batch processing on full dataset
- [ ] Verify B3 results match reported values
- [ ] Run `scripts/harden_network_stats.py` to reproduce network discoveries
