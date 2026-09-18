# Benchmark: flywing

Images: 10

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neural | 0.894 | 0.865 | 0.833 | 2.000 | 0.883 | 0.883 | 0.741 | 0.994 | 761.800 | 1.000 | 0.875 | 1.200 |

Vertex metrics compare multicellular vertices (>= 3 incident cells) derived exactly from the GT
label image with those of the reconstruction; incident-set and cyclic-order accuracy are over
matched vertices.