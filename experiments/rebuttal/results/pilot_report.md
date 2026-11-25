# Pilots beyond single-word lookups

Boundaries within ±1 layer of a top-3 shape-change peak (pooled over conditions); null = circular shift of peaks (10,000 draws).

| model | pilot | n | on peak | within ±1 | null | p |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | arith | 26 | 0.54 | **0.85** | 0.35 | 0.0026 |
| Llama-3.1-8B-Instruct | kin | 10 | 0.80 | **1.00** | 0.36 | 0.0250 |
| Qwen2.5-7B-Instruct | arith | 28 | 0.43 | **0.89** | 0.40 | 0.0006 |
| Qwen2.5-7B-Instruct | kin | 10 | 0.60 | **0.90** | 0.43 | 0.0108 |

## Llama-3.1-8B-Instruct · arith

| condition | N | acc | pairs | peaks (shape) | probe | patch | steer | handoff | onset |
|---|---|---|---|---|---|---|---|---|---|
| ops1_short | 153 | 1.00 | 120 | 18,15,13 | 16 | 16 | 18 | 16 | 18 |
| ops2_short | 154 | 0.23 | 29 | 13,15,26 | – | 16 | 18 | 16 | 26 |
| ops3_short | 155 | 0.14 | 18 | 13,16,9 | – | 16 | 17 | 16 | 19 |
| ops1_long | 160 | 0.99 | 120 | 16,18,21 | 16 | 16 | 18 | 16 | 18 |
| ops2_long | 156 | 0.11 | 13 | 13,16,9 | – | 16 | 22 | 16 | 23 |
| ops3_long | 149 | 0.11 | 16 | 13,16,8 | – | 16 | 15 | 16 | 15 |

## Llama-3.1-8B-Instruct · kin

| condition | N | acc | pairs | peaks (shape) |  peaks (by relation word) |probe | patch | steer | handoff | onset |
|---|---|---|---|---|---|---|---|---|---|---|
| hop2 | 150 | 0.87 | 120 | 17,14,12 | 14,17,31 | 13 | 17 | 17 | 17 | 17 |
| hop3 | 150 | 0.61 | 75 | 14,17,9 | 14,17,31 | 13 | 17 | 17 | 17 | 17 |

Peaks of the shape grouped by relation word instead of letter: within ±1 1.00 (null 0.38, p 0.0075).

## Qwen2.5-7B-Instruct · arith

| condition | N | acc | pairs | peaks (shape) | probe | patch | steer | handoff | onset |
|---|---|---|---|---|---|---|---|---|---|
| ops1_short | 153 | 1.00 | 120 | 23,18,14 | 23 | 22 | 23 | 22 | 23 |
| ops2_short | 154 | 0.94 | 120 | 23,14,25 | 23 | 22 | 25 | 22 | 24 |
| ops3_short | 155 | 0.67 | 95 | 14,23,27 | – | 22 | 25 | 22 | 25 |
| ops1_long | 160 | 1.00 | 120 | 23,16,18 | 23 | 22 | 23 | 22 | 23 |
| ops2_long | 156 | 0.62 | 93 | 23,16,25 | 23 | 22 | 25 | 22 | 25 |
| ops3_long | 149 | 0.26 | 29 | 23,16,14 | – | 23 | 22 | 22 | 25 |

## Qwen2.5-7B-Instruct · kin

| condition | N | acc | pairs | peaks (shape) |  peaks (by relation word) |probe | patch | steer | handoff | onset |
|---|---|---|---|---|---|---|---|---|---|---|
| hop2 | 150 | 0.91 | 120 | 19,16,24 | 19,16,24 | 16 | 24 | 24 | 20 | 22 |
| hop3 | 150 | 0.83 | 120 | 16,19,24 | 19,16,27 | 16 | 24 | 24 | 20 | 23 |

Peaks of the shape grouped by relation word instead of letter: within ±1 0.60 (null 0.43, p 0.2559).
