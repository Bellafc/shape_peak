# Grouping by the input instead of the answer

| model | task | boundaries | top-3 peaks, answer-grouped | top-3 peaks, input-grouped |
|---|---|---|---|---|
| Llama-8B | month | 7 | 18 21 16 | 18 12 14 |
| Llama-8B | weekday | 7 | 17 22 13 | 18 16 14 |
| Qwen-7B | month | 7 | 23 25 14 | 22 18 14 |
| Qwen-7B | weekday | 7 | 23 14 16 | 21 23 16 |

Exactly on a peak (within ±1); shift = circularly shifted peaks of the same grouping.

| grouping | Top-1 | Shift top-1 | Top-3 | Shift top-3 |
|---|---|---|---|---|
| answer | 0.54 (0.82) | 0.05 (0.14) | 0.64 (1.00) | 0.14 (0.37) |
| input | 0.29 (0.57) | 0.04 (0.14) | 0.54 (0.82) | 0.13 (0.36) |

## Per model and task

| model | task | n | grouping | Top-1 | Shift top-1 | Top-3 | Shift top-3 |
|---|---|---|---|---|---|---|---|
| Llama-8B | month | 7 | answer | 0.43 (0.57) | 0.04 (0.12) | 0.71 (1.00) | 0.14 (0.35) |
| Llama-8B | month | 7 | input | 0.43 (0.57) | 0.04 (0.13) | 0.43 (0.57) | 0.12 (0.34) |
| Llama-8B | weekday | 7 | answer | 0.00 (0.71) | 0.05 (0.13) | 0.14 (1.00) | 0.12 (0.36) |
| Llama-8B | weekday | 7 | input | 0.57 (0.57) | 0.04 (0.13) | 0.71 (0.71) | 0.12 (0.30) |
| Qwen-7B | month | 7 | answer | 0.86 (1.00) | 0.05 (0.15) | 0.86 (1.00) | 0.15 (0.38) |
| Qwen-7B | month | 7 | input | 0.14 (1.00) | 0.05 (0.15) | 0.14 (1.00) | 0.14 (0.44) |
| Qwen-7B | weekday | 7 | answer | 0.86 (1.00) | 0.05 (0.14) | 0.86 (1.00) | 0.14 (0.39) |
| Qwen-7B | weekday | 7 | input | 0.00 (0.14) | 0.04 (0.14) | 0.86 (1.00) | 0.15 (0.37) |
