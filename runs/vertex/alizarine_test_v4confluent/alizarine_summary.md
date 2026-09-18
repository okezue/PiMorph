# Benchmark: alizarine

Images: 10

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neural | 0.977 | 0.976 | 0.990 | 1.000 | 0.973 | 0.973 | 0.911 | 0.998 | 64.900 | 1.000 | 0.947 | 1.467 |

Vertex metrics compare multicellular vertices (>= 3 incident cells) derived exactly from the GT
label image with those of the reconstruction; incident-set and cyclic-order accuracy are over
matched vertices.