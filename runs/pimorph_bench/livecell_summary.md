# Benchmark: livecell

Images: 6

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cellpose_sam | 0.558 | 0.466 | 0.327 | 1.707 | 0.657 | 0.657 | 0.628 | 0.894 | 325.667 | 1.000 | 0.546 | 9.762 |
| neural | 0.000 | 0.000 | 0.027 | 1.414 | 0.000 | 0.000 | 0.031 | 0.460 | 750.333 | 1.000 | 0.000 | 6.906 |

Vertex metrics compare multicellular vertices (>= 3 incident cells) derived exactly from the GT
label image with those of the reconstruction; incident-set and cyclic-order accuracy are over
matched vertices.