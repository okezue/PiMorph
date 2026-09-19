# Benchmark: rpe_zo1

Images: 20

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classical | 0.527 | 0.326 | 0.251 | 2.000 | 0.562 | 0.562 | 0.505 | 0.710 | 558.050 | 1.000 | 0.538 | 0.720 |
| neural | 0.560 | 0.352 | 0.307 | 1.414 | 0.643 | 0.643 | 0.517 | 0.712 | 608.550 | 1.000 | 0.576 | 0.566 |

Vertex metrics compare multicellular vertices (>= 3 incident cells) derived exactly from the GT
label image with those of the reconstruction; incident-set and cyclic-order accuracy are over
matched vertices.