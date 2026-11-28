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
