# Peaks and boundaries from disjoint prompt halves

24 model-task pairs, 95 boundaries measured on half B. Cells: on a peak (within ±1). Null: circular shift.

| peaks from | on peak | within ±1 | null | p |
|---|---|---|---|---|
| half A (disjoint) | 0.64 | **0.86** | 0.31 | 0.0001 |
| half B (same prompts) | 0.59 | **0.87** | 0.31 | 0.0001 |

Top-3 peaks of half A within ±1 of a peak of half B: 0.78.

## By boundary

| boundary | n | half A ±1 | half B ±1 |
|---|---|---|---|
| handoff | 24 | 0.96 | 0.92 |
| onset | 23 | 0.70 | 0.78 |
| patching | 24 | 0.96 | 0.92 |
| steering | 24 | 0.83 | 0.88 |

## By model

| model | n | half A ±1 | half B ±1 |
|---|---|---|---|
| Gemma-9B | 16 | 0.88 | 0.94 |
| Llama-8B | 16 | 0.94 | 0.94 |
| Mistral-7B | 15 | 0.73 | 0.60 |
| Phi-4 | 16 | 0.62 | 0.75 |
| Qwen-14B | 16 | 1.00 | 1.00 |
| Qwen-7B | 16 | 1.00 | 1.00 |
