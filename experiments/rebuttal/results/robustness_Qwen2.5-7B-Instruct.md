# Robustness to the peak rule: Qwen2.5-7B-Instruct

top3_noThr: top-3 local maxima without the 1.5x-median threshold. argmax: the single highest late layer. contrast: median over tasks of late max / late median.

| metric | top3_noThr within ±1 | p | argmax within ±1 | argmax null | p | contrast |
|---|---|---|---|---|---|---|
| procrustes | 0.94 | 0.0001 | 0.59 | 0.14 | 0.0003 | 7.9 |
| mag_scale | 0.96 | 0.0001 | 0.71 | 0.14 | 0.0001 | 3.3 |
| mag_norm | 0.65 | 0.0313 | 0.16 | 0.14 | 0.4113 | 2.4 |
| mag_sep | 0.93 | 0.0001 | 0.53 | 0.14 | 0.0002 | 4.8 |
| mag_classnorm | 0.53 | 0.1866 | 0.00 | 0.14 | 1.0000 | 14.1 |
| grid_c1_s1_a1 | 0.94 | 0.0001 | 0.59 | 0.14 | 0.0001 | 7.9 |
| grid_c1_s1_a0 | 0.69 | 0.0071 | 0.50 | 0.14 | 0.0006 | 1.4 |
| grid_c1_s0_a1 | 0.96 | 0.0001 | 0.71 | 0.14 | 0.0001 | 6.6 |
| grid_c1_s0_a0 | 0.60 | 0.0441 | 0.31 | 0.14 | 0.0512 | 1.3 |
| grid_c0_s1_a1 | 0.79 | 0.0013 | 0.37 | 0.14 | 0.0170 | 357.8 |
| grid_c0_s1_a0 | 0.71 | 0.0136 | 0.00 | 0.14 | 1.0000 | 4.4 |
| grid_c0_s0_a1 | 0.69 | 0.0089 | 0.41 | 0.14 | 0.0046 | 4.7 |
| grid_c0_s0_a0 | 0.68 | 0.0253 | 0.00 | 0.14 | 1.0000 | 5.9 |
| dir_centroid | 0.69 | 0.0089 | 0.50 | 0.14 | 0.0009 | 1.6 |
| dir_pair | 0.69 | 0.0096 | 0.50 | 0.14 | 0.0008 | 1.6 |
