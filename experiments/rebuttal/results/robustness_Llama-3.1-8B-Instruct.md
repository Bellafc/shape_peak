# Robustness to the peak rule: Llama-3.1-8B-Instruct

top3_noThr: top-3 local maxima without the 1.5x-median threshold. argmax: the single highest late layer. contrast: median over tasks of late max / late median.

| metric | top3_noThr within ±1 | p | argmax within ±1 | argmax null | p | contrast |
|---|---|---|---|---|---|---|
| procrustes | 0.79 | 0.0002 | 0.31 | 0.12 | 0.0132 | 7.2 |
| mag_scale | 0.79 | 0.0001 | 0.35 | 0.12 | 0.0033 | 5.1 |
| mag_norm | 0.56 | 0.0285 | 0.21 | 0.12 | 0.1412 | 2.4 |
| mag_sep | 0.81 | 0.0001 | 0.35 | 0.12 | 0.0045 | 10.0 |
| mag_classnorm | 0.09 | 0.9976 | 0.03 | 0.12 | 0.9334 | 4.0 |
| grid_c1_s1_a1 | 0.79 | 0.0001 | 0.31 | 0.12 | 0.0117 | 7.2 |
| grid_c1_s1_a0 | 0.59 | 0.0085 | 0.19 | 0.12 | 0.1870 | 2.5 |
| grid_c1_s0_a1 | 0.81 | 0.0001 | 0.35 | 0.12 | 0.0038 | 13.0 |
| grid_c1_s0_a0 | 0.26 | 0.6566 | 0.10 | 0.12 | 0.5978 | 2.6 |
| grid_c0_s1_a1 | 0.74 | 0.0002 | 0.44 | 0.12 | 0.0001 | 14.0 |
| grid_c0_s1_a0 | 0.15 | 0.9629 | 0.00 | 0.12 | 1.0000 | 3.8 |
| grid_c0_s0_a1 | 0.66 | 0.0018 | 0.28 | 0.12 | 0.0263 | 5.6 |
| grid_c0_s0_a0 | 0.16 | 0.9601 | 0.00 | 0.12 | 1.0000 | 4.7 |
| dir_centroid | 0.59 | 0.0083 | 0.19 | 0.12 | 0.1824 | 3.4 |
| dir_pair | 0.59 | 0.0064 | 0.19 | 0.12 | 0.1860 | 3.4 |
