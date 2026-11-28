# Pilots: peak controls and search cost

24 conditions, 111 boundaries.

## 1. Boundaries exactly on a peak (within ±1)

| Peaks from | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Own | 0.22 (0.34) | 0.52 (0.85) | 0.55 (0.89) |
| Other task, same model | 0.04 (0.23) | 0.10 (0.62) | 0.25 (0.81) |
| Other model (relative depth) | 0.01 (0.11) | 0.07 (0.36) | 0.18 (0.55) |
| Circular shift | 0.04 (0.12) | 0.12 (0.35) | 0.18 (0.50) |

## 2. Layers tested to find the boundary (mean)

| Strategy | handoff (n=24) | onset (α=1) (n=24) | correct |
|---|---|---|---|
| Full sweep | 25.0 | 25.0 | 1.00 |
| Binary search | 4.8 | 4.7 | 0.98 |
| Peaks (optimal tree, fixed prior) | 3.4 | 4.6 | 1.00 |

### By model and pilot (mean over both events)

| model | pilot | full sweep | binary | peaks |
|---|---|---|---|---|
| Llama-8B | arith | 24.0 | 4.8 | 3.9 |
| Llama-8B | kin | 24.0 | 4.0 | 2.5 |
| Phi-4 | arith | 30.0 | 5.0 | 5.1 |
| Phi-4 | kin | 30.0 | 5.0 | 3.0 |
| Qwen-7B | arith | 21.0 | 4.8 | 3.6 |
| Qwen-7B | kin | 21.0 | 4.2 | 4.8 |
