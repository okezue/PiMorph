# Models

The proposal checkpoints are not in this repository. They live in the PiMorph data record on
Zenodo (concept DOI in `scripts/fetch_zenodo.py`; it always resolves to the newest version) together
with the training logs and configurations. Fetch them with

```bash
python scripts/fetch_zenodo.py --models        # downloads pimorph_models.tar, verifies the md5, extracts into models/
```

| Checkpoint | Size | What it is | Card |
|---|---|---|---|
| `pimorph_proposals_v6_pool.pt` | 78 MB | recommended for confluent monolayers with a bright membrane or junction channel; trained for topologically missed vertices, pooled hCEC, alizarine, FlyWing, RPE and real PECAM-1 HUVEC pseudo-labels | `pimorph_proposals_v6_pool.md` |
| `pimorph_proposals_v4_confluent.pt` | 78 MB | first in-domain fine-tune on the confluent truth sets | `pimorph_proposals_v4_confluent.md` |
| `pimorph_proposals_v3_endo.pt` | 78 MB | v2 recipe on the corrected HAEC reference; best choice for small-cell E-cadherin epithelia before v6 | `pimorph_proposals_v3_endo.md` |
| `pimorph_proposals_v2_endo.pt` | 78 MB | first model trained on endothelial truth (HAEC, mCellSeg) | `pimorph_proposals_v2_endo.md` |
| `pimorph_proposals_v1_multi.pt` | 78 MB | synthetic + LIVECell + NeurIPS CellSeg | `pimorph_proposals_v1_multi.md` |
| `pimorph_proposals_v0_mixed.pt`, `pimorph_proposals_v0_synth.pt` | 35 MB each | first checkpoints, synthetic tissues with and without pseudo-labels | `pimorph_proposals_v0_*.md` |

`ajmorph_classifier*.joblib` (the legacy junction morphology classifier) stays here; it is small
and the legacy pipeline loads it by path.

`git` ignores `models/*.pt`, so a fetched checkpoint is never committed by accident.
