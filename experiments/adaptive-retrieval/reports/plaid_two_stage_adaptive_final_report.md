# PLAID two-stage adaptive runtime and signal report

## Scope

45 frozen queries (old=9; independent=36), 3 repetitions, existing indexes only. The preceding six-query parity gate passed 18/18. This report contains a one-exact-finalize two-stage API: 400/512 checkpoints are approximate-only. No index build or parameter sweep was run.

## Approximate signal stability

The rule feature was selected using old-set budget-400 AUC only: **approx_cutoff_margin**. Its cutoff direction and thresholds were then frozen from old queries for both 400 and 512 decisions. Independent labels were not used to choose the feature or thresholds.

- Old: truncation-risk queries at 400 = 3/9; candidate-generation-failure queries = 0/9; mean candidate-generation Recall ceiling = 1.0000.
- Independent: truncation-risk queries at 400 = 9/36; candidate-generation-failure queries = 5/36; mean candidate-generation Recall ceiling = 0.9489.

Feature AUC/effect summaries are in `plaid_approx_signal_stability.csv`. Direction agreement must be judged by comparing old and independent direction for each feature/budget; oriented AUC alone can conceal a reversal.

## Policy results

| Dataset | Policy | Mean ms | P50 ms | P95 ms | QPS* | Recall@100 | Recall@10 | MRR@10 | Success@10 | >400 fraction | Candidate-gen failure | Candidate-gen ceiling |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old | Always400 | 27.033 | 25.945 | 36.426 | 36.991 | 0.763 | 0.507 | 0.926 | 1.000 | 0.000 | 0.000 | 1.000 |
| old | Always512 | 27.500 | 26.105 | 37.667 | 36.364 | 0.834 | 0.507 | 0.926 | 1.000 | 1.000 | 0.000 | 1.000 |
| old | Always1024 | 28.136 | 26.842 | 38.124 | 35.542 | 0.871 | 0.507 | 0.926 | 1.000 | 1.000 | 0.000 | 1.000 |
| old | TwoStageAdaptive_frozen_old_thresholds | 30.996 | 29.234 | 42.090 | 32.263 | 0.871 | 0.507 | 0.926 | 1.000 | 0.556 | 0.000 | 1.000 |
| old | Oracle_upper_bound | 30.106 | 29.043 | 42.090 | 33.216 | 0.871 | 0.507 | 0.926 | 1.000 | 0.333 | 0.000 | 1.000 |
| independent | Always400 | 27.775 | 27.490 | 31.532 | 36.004 | 0.850 | 0.649 | 0.800 | 0.944 | 0.000 | 0.139 | 0.949 |
| independent | Always512 | 27.878 | 27.589 | 32.192 | 35.871 | 0.875 | 0.649 | 0.800 | 0.944 | 1.000 | 0.139 | 0.949 |
| independent | Always1024 | 28.578 | 28.238 | 32.719 | 34.991 | 0.908 | 0.649 | 0.800 | 0.944 | 1.000 | 0.139 | 0.949 |
| independent | TwoStageAdaptive_frozen_old_thresholds | 31.405 | 30.854 | 34.547 | 31.842 | 0.886 | 0.649 | 0.800 | 0.944 | 0.722 | 0.139 | 0.949 |
| independent | Oracle_upper_bound | 29.895 | 28.856 | 34.191 | 33.450 | 0.908 | 0.649 | 0.800 | 0.944 | 0.250 | 0.139 | 0.949 |

*QPS is 1000 / mean single-query end-to-end latency, not batched throughput.

## Interpretation

Two-stage runtime includes query encoding, candidate generation, all approximate checkpoints used, decision/state work, one final exact decode/MaxSim, sorting and external CUDA synchronization. GPU-event sub-timings are reported separately in the raw runtime CSV; full legacy score_pids is reported as a unified filter+refine+sort interval because it has no internal event instrumentation.

Candidate-generation failures are separate from truncation risk. The per-query candidate-generation recall ceiling is recorded in the feature CSV; ndocs adaptation cannot recover relevant PIDs absent from the generated candidate set. The oracle policy is only an upper bound because it uses qrels and full-budget outcomes.

## Generated files

- `plaid_two_stage_runtime.csv` — per-query/repetition latency and stage timings.
- `plaid_approx_dynamic_features.csv` — query-level approximate signals and truncation/candidate-generation labels.
- `plaid_approx_signal_stability.csv` — old/independent AUC, medians, effect sizes and direction.
- `plaid_two_stage_policy_results.csv` — Always400/512/1024, frozen-old adaptive and qrels oracle summaries.
