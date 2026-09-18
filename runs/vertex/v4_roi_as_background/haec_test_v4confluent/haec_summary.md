# Benchmark: haec

Images: 86

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neural | 0.518 | 0.328 | 0.297 | 1.414 | 0.702 | 0.702 | 0.609 | 0.851 | 1244.419 | 1.000 | 0.507 | 1.633 |

Vertex metrics compare multicellular vertices (>= 3 incident cells) derived exactly from the GT
label image with those of the reconstruction; incident-set and cyclic-order accuracy are over
matched vertices.