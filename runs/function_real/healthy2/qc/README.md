# QC pack: QBAM crops with decoded outlines

12 well-timepoints stratified over TER (see `qc_index.csv`). Each PNG (900 x 480) shows a centre crop
of the blue absorbance tile sized to hold about 25 cells on the left and, on the right, the same crop
with the decoded cell outlines (yellow) and multicellular vertices (cyan circles). Bright = high
absorbance (pigment); cell borders are the thin darker lines. The title gives plate, well, condition,
week and the measured TER.

How to record judgments: fill `qc_judgments.csv` in this folder with one row per PNG and the columns

    tile, cells_visible, cells_predicted, merges, splits, verdict

- `tile`: the PNG file name.
- `cells_visible`: your count of cells in the LEFT panel (count a cell if more than half of it is
  inside the crop).
- `cells_predicted`: your count of outlined cells in the RIGHT panel with the same rule (the title
  reports the number of label ids touching the crop, which is larger).
- `merges`: outlined regions that clearly contain two or more real cells.
- `splits`: real cells cut into two or more outlined regions.
- `verdict`: `good` (merges + splits <= 2), `usable` (3 to 5) or `bad` (more than 5, or outlines
  unrelated to the visible borders).

The segmenter was fine-tuned on mature, well pigmented AMD tiles; the weeks 3 to 5 crops (low TER)
are less pigmented and are where failures are expected. Please note in `verdict` if borders are not
visible to you either.
