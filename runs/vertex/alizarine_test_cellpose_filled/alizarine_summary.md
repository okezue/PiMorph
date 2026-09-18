# Benchmark: alizarine

Images: 10

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cellpose_sam_filled | 0.989 | 0.989 | 0.984 | 1.000 | 0.991 | 0.991 | 0.898 | 1.000 | 30.000 | 1.000 | 0.952 | 1.554 |

Vertex metrics compare multicellular vertices (>= 3 incident cells) derived exactly from the GT
label image with those of the reconstruction; incident-set and cyclic-order accuracy are over
matched vertices.