# Robustness to the peak rule: Llama-3.1-8B-Instruct

top3_noThr: top-3 local maxima without the 1.5x-median threshold. argmax: the single highest late layer. contrast: median over tasks of late max / late median.

| metric | top3_noThr within ±1 | p | argmax within ±1 | argmax null | p | contrast |
|---|---|---|---|---|---|---|
| procrustes | 0.82 | 0.0002 | 0.32 | 0.13 | 0.0132 | 7.2 |
| mag_scale | 0.82 | 0.0001 | 0.36 | 0.12 | 0.0033 | 5.1 |
| mag_norm | 0.58 | 0.0285 | 0.21 | 0.12 | 0.1412 | 2.4 |
| mag_sep | 0.83 | 0.0001 | 0.36 | 0.13 | 0.0045 | 10.0 |
| mag_classnorm | 0.09 | 0.9976 | 0.03 | 0.12 | 0.9334 | 4.0 |
| grid_c1_s1_a1 | 0.82 | 0.0001 | 0.32 | 0.12 | 0.0117 | 7.2 |
| grid_c1_s1_a0 | 0.61 | 0.0085 | 0.20 | 0.12 | 0.1870 | 2.5 |
| grid_c1_s0_a1 | 0.83 | 0.0001 | 0.36 | 0.13 | 0.0038 | 13.0 |
| grid_c1_s0_a0 | 0.27 | 0.6566 | 0.11 | 0.13 | 0.5978 | 2.6 |
| grid_c0_s1_a1 | 0.76 | 0.0002 | 0.45 | 0.12 | 0.0001 | 14.0 |
| grid_c0_s1_a0 | 0.15 | 0.9629 | 0.00 | 0.12 | 1.0000 | 3.8 |
| grid_c0_s0_a1 | 0.68 | 0.0018 | 0.29 | 0.12 | 0.0263 | 5.6 |
| grid_c0_s0_a0 | 0.17 | 0.9601 | 0.00 | 0.13 | 1.0000 | 4.7 |
| dir_centroid | 0.61 | 0.0083 | 0.20 | 0.12 | 0.1824 | 3.4 |
| dir_pair | 0.61 | 0.0064 | 0.20 | 0.12 | 0.1860 | 3.4 |
