# Robustness of the comparison to the peak rule (Qwen2.5-7B-Instruct, half prompt set)

top3_noThr: top-3 local maxima from layer 7 without the 1.5x-median height threshold.
argmax: the single highest late layer only. contrast: median over tasks of (late-layer max / late-layer median).
p: circular-shift null, 10,000 draws (computed locally from rebuttal_results/metrics with experiments/rebuttal/evaluate.py functions).

| metric | top3_noThr within ±1 | p | argmax within ±1 | argmax null | p | contrast |
|---|---|---|---|---|---|---|
| procrustes | 0.94 | <0.001 | 0.59 | 0.14 | <0.001 | 7.9 |
| mag_scale | 0.96 | <0.001 | 0.71 | 0.14 | <0.001 | 3.3 |
| mag_norm | 0.65 | 0.031 | 0.16 | 0.14 | 0.41 | 2.4 |
| mag_sep | 0.93 | <0.001 | 0.53 | 0.14 | <0.001 | 4.8 |
| mag_classnorm | 0.53 | 0.18 | 0.00 | 0.14 | 1.00 | 14.1 |
| grid_c1_s1_a1 | 0.94 | <0.001 | 0.59 | 0.14 | <0.001 | 7.9 |
| grid_c1_s1_a0 | 0.69 | 0.008 | 0.50 | 0.14 | <0.001 | 1.4 |
| grid_c1_s0_a1 | 0.96 | <0.001 | 0.71 | 0.14 | <0.001 | 6.6 |
| grid_c1_s0_a0 | 0.60 | 0.041 | 0.31 | 0.14 | 0.049 | 1.3 |
| grid_c0_s1_a1 | 0.79 | 0.001 | 0.37 | 0.14 | 0.012 | 357.8 |
| grid_c0_s1_a0 | 0.71 | 0.017 | 0.00 | 0.14 | 1.00 | 4.4 |
| grid_c0_s0_a1 | 0.69 | 0.011 | 0.41 | 0.14 | 0.006 | 4.7 |
| grid_c0_s0_a0 | 0.68 | 0.026 | 0.00 | 0.14 | 1.00 | 5.9 |
| dir_centroid | 0.69 | 0.009 | 0.50 | 0.14 | <0.001 | 1.6 |
| dir_pair | 0.69 | 0.009 | 0.50 | 0.14 | 0.001 | 1.6 |
