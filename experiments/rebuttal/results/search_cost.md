# Search cost per causal boundary

Layers tested (mean) and share of pairs where the strategy returns the full-sweep layer.


## 1 handoff (53 pairs; late layers 27.9)

| strategy | layers tested | correct |
|---|---|---|
| full sweep | 27.9 | 1.00 |
| binary search | 5.8 | 1.00 |
| start at logit lens | 4.1 | 1.00 |
| start at highest peak | 3.7 | 1.00 |

Peak start cheaper than logit-lens start in 0.40 of pairs, dearer in 0.28.

## 2 onset (α=1) (59 pairs; late layers 27.9)

| strategy | layers tested | correct |
|---|---|---|
| full sweep | 27.9 | 1.00 |
| binary search | 5.8 | 1.00 |
| start at logit lens | 3.8 | 1.00 |
| start at highest peak | 3.9 | 1.00 |

Peak start cheaper than logit-lens start in 0.41 of pairs, dearer in 0.34.

## 3 transfer (T ≥ .5) (58 pairs; late layers 27.8)

| strategy | layers tested | correct | interventions |
|---|---|---|---|
| full sweep | 27.8 | 1.00 | 290 |
| binary search | 5.8 | 1.00 | 60 |
| start at logit lens | 3.4 | 1.00 | 36 |
| start at highest peak | 4.6 | 1.00 | 46 |

Peak start cheaper than logit-lens start in 0.26 of pairs, dearer in 0.47.
