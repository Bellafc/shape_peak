# Rebuttal: alternative shape metrics

Do the shape-change peaks depend on the Procrustes choice? We recompute the adjacent-layer change of the answer-class
configuration with other metrics and test whether their peaks still locate the boundaries measured in the paper.
Only forward passes are needed: the boundary layers (probe, patching and steering rises, handoff, steering onset at α = 1 and 8,
transfer rise) come from the release data and do not depend on the shape metric.

Setup: Qwen2.5-7B-Instruct, all ten tasks, half prompt set (`SUB=1`: k = 1, 3, 5 for arithmetic; the first half of the cues for retrieval).

| Group | Metrics | Reviewer point |
|---|---|---|
| Reference | `procrustes` (Eq. 1) | |
| A. Magnitude only | `mag_scale` (scale removed by Eq. 1), `mag_norm` (residual norm), `mag_sep` (class spread / residual norm), `mag_classnorm` (class-specific norm change) | R1: can magnitude changes alone identify the peaks? |
| C6. Normalisation ablation | `grid_c{0,1}_s{0,1}_a{0,1}`: centring × unit norm × orthogonal alignment; `grid_c0_s0_a0` = raw relative displacement | W5 (centring, unit norm), W3 (alignment) |
| C7. Raw-coordinate directions | `dir_centroid`, `dir_pair` (centroid-difference = steering directions, no alignment) | W3: orientation in the original coordinates |

Exact definitions are in the docstring of `alt_metrics.py`.

## Run (one A40, ~15 min)

```bash
bash experiments/rebuttal/run.sh
```

Steps: `extract_acts.py` (GPU, per task) → `alt_metrics.py` (CPU) → `evaluate.py` (CPU: peaks with the paper's rule, within ±1 of each
boundary, circular-shift null with 10,000 draws). Output: `rebuttal_results/report_Qwen2.5-7B-Instruct.md`, `curves_*.png`, per-layer CSVs.
