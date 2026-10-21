# CPU online replay and supplemental query audit

## What was run

- Frozen stagewise LightGBM controller; thresholds were not changed.
- CPU only (`PLAID_DEVICE=cpu`, `PLAID_SIGNAL_DEVICE=cpu`).
- Original evaluation: the locked 42-query test split. Query labels were used only after retrieval to calculate quality.
- Supplemental cohort: LOTTE pooled/test; 100 fixed-seed random queries plus 50 randomly selected queries from the remaining top query-length quartile. Seed: `20261016`. Exact duplicate query text from the locked replay was excluded. The length-enriched cohort is reported separately and is not a population estimate.
- Supplemental profiles are external evaluation data. Do not use them for training or threshold selection while reporting results on this pooled/test cohort.

## Execution-path checks

The measured online loop encodes once, generates candidates once, passes the same `previous_state` through successive approximate checkpoints, calls the frozen model only at reached 400/512/768 boundaries, then runs exact finalization once.

| Cohort | Queries | Reached 400 | Reached 512 | Reached 768 | Reached 1024 | Exact finalize once | IDs / calls validated |
|---|---:|---:|---:|---:|---:|---:|---|
| Locked lifestyle test | 42 | 42 | 24 | 10 | 0 | 42/42 | Pass |
| Pooled external sample | 150 | 150 | 68 | 12 | 0 | 150/150 | Pass |

All measured queries had exactly one encode, one candidate-generation call, and one exact-finalize call. Online predicted budgets matched the earlier offline model predictions for all 42 locked test queries. The warmup path traversed through 1024, but no measured query caused the learned model to continue from 768 to 1024; therefore that controller-positive branch has not yet been observed in a measured request.

## Locked 42-query test result

| Method | Mean Recall@100 | Mean end-to-end latency | Median latency |
|---|---:|---:|---:|
| Learned online adaptive | 0.6774 | 195.1 ms | 189.0 ms |
| Fixed1024 matched CPU reference | 0.7091 | 121.6 ms | 120.2 ms |

The learned policy selected 400 for 18 queries, 512 for 14, 768 for 10, and 1024 for none. This is an execution-validation result, not a tuning run. It currently shows neither a latency win nor quality parity against fixed1024.

## Supplemental pooled/test result

Oracle labels use the predeclared rule: choose the lowest-latency budget whose Recall@100 is at least `alpha × fixed1024 Recall@100`. With the measured fixed-budget profiles, alpha 0.95 and 0.98 produced the same distribution:

| Cohort | 400 | 512 | 768 | 1024 | Count needing ≥768 |
|---|---:|---:|---:|---:|---:|
| Uniform random (n=100) | 36 | 52 | 8 | 4 | 12 |
| Length-enriched (n=50) | 16 | 26 | 5 | 3 | 8 |
| Combined (n=150; not a prevalence estimate) | 52 | 78 | 13 | 7 | 20 |

The frozen controller selected 400/512/768/1024 for 82/56/12/0 queries respectively. Only 1 of the 20 queries whose oracle budget was at least 768 was assigned a budget of at least 768; none of the seven oracle-1024 queries was assigned 1024. This is evidence that the current controller misses high-budget cases on this external domain, not evidence to tune it on this test cohort.

Because the oracle rule chooses the fastest eligible budget and this supplemental profile has one latency measurement per budget, its labels can be affected by timing noise. As a quality-only cross-check, the *minimum* budget satisfying the same recall threshold was 400/512/768/1024 for 133/5/8/4 queries (random group: 93/1/4/2; length-enriched group: 40/4/4/2). Treat the latency-minimizing oracle distribution as exploratory until repeated latency runs are collected.

| Pooled/test comparison | Mean Recall@100 | Mean latency |
|---|---:|---:|
| Fixed400 | 0.6551 | 228.0 ms |
| Fixed512 | 0.6635 | 227.2 ms |
| Fixed768 | 0.6870 | 238.0 ms |
| Fixed1024 | 0.6921 | 249.3 ms |
| Learned online adaptive | 0.6605 | 469.9 ms |

The adaptive online run is slower than fixed1024 in this measurement and has lower mean recall. Controller inference itself is small; the accumulated CPU approximate stages dominate the extra work. Do not claim a speedup from these results.

## Sampling/training boundary

The original LOTTE lifestyle-dev query set contains 417 queries and is already fully profiled. The server has a matching `plaid-pooled-test-nbits2` index for the external cohort, but the inspected LOTTE pooled-dev query pool (2,931 queries) does not currently have a matching PLAID index. Therefore the 150 new queries are correctly treated as external evaluation/diagnostic data, not extra training examples.

To add high-budget training examples without contaminating evaluation, the next data prerequisite is a matching indexed, qrel-backed development/training corpus (for example, pooled-dev). After that index is available, use a fixed-seed random cohort plus a separately reported query-length-enriched challenge cohort; profile every selected query at all four budgets, keep all outcomes, and train only on that development data. Preserve both current test cohorts and all thresholds untouched until the evaluation protocol is locked.

## Artifacts

- `online_test_replay.jsonl`, `online_test_fixed1024_reference.jsonl`
- `supplemental_sample_manifest.json`, `supplemental_budget_profiles.jsonl`, `supplemental_online_replay.jsonl`
- `online_test_replay_summary.json`, `online_test_fixed1024_reference.jsonl.summary.json`, `supplemental_budget_profiles_summary.json`, `supplemental_online_replay_summary.json`
