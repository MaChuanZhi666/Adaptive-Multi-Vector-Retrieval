# Final benchmark numbers

## Four benchmark settings

| Setting | Mean latency (ms/query) | p50 (ms) | p95 (ms) | QPS | Success@10 | MRR@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PLAID-GPU | 28.670 | 28.096 | 31.690 | 34.880 | 0.9167 | 0.7648 | 0.7108 |
| PLAID-CPU | 364.717 | 290.967 | 576.915 | 2.742 | 0.9167 | 0.7648 | 0.7108 |
| WARP-GPU | 83.401 | 72.480 | 145.809 | 11.990 | 0.9167 | 0.7478 | 0.6694 |
| WARP-CPU | 673.524 | 617.788 | 1076.633 | 1.485 | 0.9167 | 0.7500 | 0.6590 |

## Extrema and speedup

- Minimum mean latency: PLAID-GPU, 28.670 ms/query.
- Maximum mean latency: WARP-CPU, 673.524 ms/query.
- Maximum QPS: PLAID-GPU, 34.880.
- PLAID CPU/GPU latency ratio: 12.721×.
- WARP CPU/GPU latency ratio: 8.076×.

## PLAID stage latency

| Stage | Mean (ms) | Share of end-to-end latency |
|---|---:|---:|
| Prune/filter | 7.670 | 26.17% |
| Decompress | 4.618 | 15.75% |
| MaxSim/GEMM | 1.363 | 4.65% |
| Sorting | 0.159 | 0.54% |


- Instrumented PLAID end-to-end mean: 29.313 ms/query.
- `score_pids` total mean: 13.653 ms/query.
- Within `score_pids`: prune/filter 56.18%, decompress 33.82%, MaxSim/GEMM 9.98%.
- Sorting is measured after `score_pids` returns and is not part of `score_pids_total_ms`.
