# Benchmark: flywing

Images: 10

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cellpose_sam_filled | 0.966 | 0.940 | 0.849 | 1.414 | 0.973 | 0.973 | 0.795 | 0.996 | 302.000 | 1.000 | 0.926 | 1.016 |

Vertex metrics compare multicellular vertices (>= 3 incident cells) derived exactly from the GT
label image with those of the reconstruction; incident-set and cyclic-order accuracy are over
matched vertices.