| Test split | Method | Adj F1 pair | Vertex F1 | Vertex prec. / rec. | pred / true cells | pred / true vertices | Loc. median (px) | PQ | Boundary F1 |
|---|---|---|---|---|---|---|---|---|---|
| hCEC (5) | Cellpose-SAM zero-shot | 0.830 | 0.635 | 0.721 / 0.581 | 1635 / 1665 | 2298 / 2974 | 1.17 | 0.790 | 0.880 |
| hCEC (5) | Cellpose-SAM ft confluent pool | 0.937 | 0.695 | 0.691 / 0.700 | 1705 / 1665 | 3023 / 2974 | 1.40 | 0.862 | 0.924 |
| hCEC (5) | Cellpose-SAM ft hCEC only | 0.937 | 0.696 | 0.691 / 0.701 | 1705 / 1665 | 3025 / 2974 | 1.40 | 0.862 | 0.924 |
| hCEC (5) | Cellpose-SAM ft HAEC (cross-domain) | 0.207 | 0.098 | 0.258 / 0.077 | 1151 / 1665 | 509 / 2974 | 2.05 | 0.264 | 0.525 |
| hCEC (5) | PiMorph v6_pool tuned | 0.867 | 0.680 | 0.678 / 0.683 | 1685 / 1665 | 3014 / 2974 | 1.28 | 0.816 | 0.916 |
| alizarine (10) | Cellpose-SAM zero-shot | 0.989 | 0.984 | 0.979 / 0.989 | 325 / 320 | 576 / 570 | 1.00 | 0.898 | 1.000 |
| alizarine (10) | Cellpose-SAM ft confluent pool | 0.992 | 0.991 | 0.986 / 0.996 | 325 / 320 | 576 / 570 | 1.00 | 0.903 | 1.000 |
| alizarine (10) | PiMorph v6_pool tuned | 0.987 | 0.992 | 0.994 / 0.990 | 320 / 320 | 568 / 570 | 1.00 | 0.920 | 0.999 |
| FlyWing (10) | Cellpose-SAM zero-shot | 0.966 | 0.849 | 0.834 / 0.864 | 711 / 742 | 1306 / 1260 | 1.47 | 0.795 | 0.996 |
| FlyWing (10) | Cellpose-SAM ft confluent pool | 0.966 | 0.852 | 0.837 / 0.868 | 712 / 742 | 1308 / 1260 | 1.53 | 0.793 | 0.993 |
| FlyWing (10) | PiMorph v6_pool tuned | 0.911 | 0.875 | 0.869 / 0.881 | 684 / 742 | 1278 / 1260 | 1.59 | 0.780 | 0.994 |
| RPE (20) | Cellpose-SAM zero-shot | 0.622 | 0.287 | 0.254 / 0.331 | 177 / 140 | 269 / 209 | 1.77 | 0.548 | 0.686 |
| RPE (20) | Cellpose-SAM ft confluent pool | 0.681 | 0.336 | 0.301 / 0.383 | 168 / 140 | 260 / 209 | 1.70 | 0.612 | 0.751 |
| RPE (20) | PiMorph v6_pool tuned | 0.605 | 0.327 | 0.291 / 0.381 | 171 / 140 | 265 / 209 | 1.73 | 0.556 | 0.731 |
| HAEC (86) | Cellpose-SAM zero-shot | 0.314 | 0.039 | 0.039 / 0.043 | 753 / 569 | 110 / 92 | 1.40 | 0.465 | 0.708 |
| HAEC (86) | Cellpose-SAM ft HAEC train | 0.555 | 0.331 | 0.371 / 0.304 | 623 / 569 | 78 / 92 | 1.46 | 0.648 | 0.862 |
| HAEC (86) | Cellpose-SAM ft confluent pool (cross) | 0.162 | 0.018 | 0.010 / 0.103 | 802 / 569 | 996 / 92 | 1.64 | 0.324 | 0.605 |
| HAEC (86) | PiMorph v6_pool tuned | 0.536 | 0.352 | 0.293 / 0.453 | 648 / 569 | 151 / 92 | 1.25 | 0.626 | 0.858 |
