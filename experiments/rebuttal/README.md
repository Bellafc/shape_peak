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
boundary, circular-shift null with 10,000 draws). Output: `rebuttal_results/` (untracked on the GPU machine); the committed copies are in `experiments/rebuttal/results/`.

## Pilots beyond single-word lookups

`pilot.py` (GPU) and `pilot_eval.py` (CPU), run by `run.sh` after the metric comparison, on Qwen2.5-7B and Llama-3.1-8B:

- `arith`: multi-step arithmetic word problems (1, 2 or 3 operations; answer 0–9), short (~60 tokens) or long (24 unrelated
  sentences interleaved, ~300 tokens).
- `kin`: multiple-choice kinship reasoning (CLUTRR-style, 2 or 3 hops); the answer class is the option letter A–D, options shuffled.

Per layer: Procrustes shape change, identity probe, patching of the user message / all answer positions / the last position,
α = 1 centroid-difference steering. Boundaries and peaks as in the paper; `pilot_report.md` tests them with the circular-shift null.
