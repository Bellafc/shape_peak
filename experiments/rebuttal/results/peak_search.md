# Search with peaks and their neighbours (layers tested, mean)

| strategy | 1 handoff | 2 onset (α=1) | 3 transfer (T ≥ .5) | mean | correct |
|---|---|---|---|---|---|
| walk from λ1 | 3.66 | 3.88 | 4.62 | 4.05 | 1.00 |
| walk from logit lens | 4.08 | 3.85 | 3.45 | 3.79 | 1.00 |
| binary search | 4.89 | 4.85 | 4.84 | 4.86 | 1.00 |
| peak pairs | 3.96 | 4.17 | 3.84 | 3.99 | 1.00 |
| prior search (fixed) | 3.57 | 3.88 | 4.07 | 3.84 | 1.00 |
| prior search (learned) | 3.74 | 3.76 | 3.98 | 3.83 | 1.00 |
| prior search (learned) + logit lens | 3.32 | 3.56 | 3.48 | 3.45 | 1.00 |
| optimal tree, prior (fixed) | 3.62 | 3.81 | 3.97 | 3.80 | 1.00 |
| optimal tree, prior (learned) | 3.72 | 3.78 | 3.84 | 3.78 | 1.00 |
| optimal tree, prior (learned) + logit lens | 3.34 | 3.53 | 3.38 | 3.41 | 1.00 |
| optimal tree, logit lens only (learned) | 4.06 | 4.12 | 3.57 | 3.91 | 1.00 |
| floor: boundary known in advance | 2.00 | 2.00 | 2.00 | 2.00 | 1.00 |
