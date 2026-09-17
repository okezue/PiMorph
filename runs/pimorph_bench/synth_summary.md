# Benchmark: synth

Images: 40

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classical | 0.447 | 0.362 | 0.503 | 1.000 | 0.372 | 0.372 | 0.554 | 0.852 | 879.750 | 1.000 | 0.434 | 4.529 |
| neural | 0.889 | 0.739 | 0.733 | 1.000 | 0.860 | 0.860 | 0.891 | 0.981 | 398.425 | 1.000 | 0.854 | 3.575 |

Vertex metrics compare multicellular vertices (>= 3 incident cells) derived exactly from the GT
label image with those of the reconstruction; incident-set and cyclic-order accuracy are over
matched vertices.