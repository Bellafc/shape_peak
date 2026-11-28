# Steering with the parts of the centroid difference

Qwen2.5-7B and Llama-3.1-8B, weekday and month; late layers (from round(0.25 L)); all answer positions.

| model | task | prompts | top-3 peaks (λ1 first) |
|---|---|---|---|
| Llama-8B | month | 212 | 18 21 16 |
| Llama-8B | weekday | 107 | 17 22 13 |
| Qwen-7B | month | 216 | 23 25 14 |
| Qwen-7B | weekday | 123 | 23 16 12 |

## Success, by layer relative to the highest peak (mean over layers and pairs)

| direction | before λ1 | at / after λ1 |
|---|---|---|
| full centroid difference | 0.00 | 0.83 |
| answer part U | 0.00 | 0.81 |
| blurred part B | 0.00 | 0.01 |
| U, rescaled to ‖full‖ | 0.00 | 0.82 |
| B, rescaled to ‖full‖ | 0.00 | 0.09 |
| random, ‖full‖ | 0.00 | 0.00 |

## Per pair (at / after λ1)

| pair | full | answer | blur | answer_eq | blur_eq | random |
|---|---|---|---|---|---|---|
| Llama-8B month | 0.84 | 0.81 | 0.01 | 0.81 | 0.09 | 0.00 |
| Llama-8B weekday | 0.73 | 0.71 | 0.01 | 0.73 | 0.10 | 0.01 |
| Qwen-7B month | 0.99 | 0.98 | 0.00 | 0.98 | 0.01 | 0.00 |
| Qwen-7B weekday | 0.96 | 0.96 | 0.00 | 0.96 | 0.16 | 0.00 |

## Layers where the full difference switches ≥ half of the prompts

| direction | share of those layers where it also switches ≥ half |
|---|---|
| answer part U | 1.00 (of 32) |
| blurred part B | 0.00 (of 32) |
| U, rescaled to ‖full‖ | 1.00 (of 32) |
| B, rescaled to ‖full‖ | 0.00 (of 32) |
| random, ‖full‖ | 0.00 (of 32) |
