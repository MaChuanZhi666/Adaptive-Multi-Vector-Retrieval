# Controller pre-training audit

Date: 2026-10-15

## Verdict

**Not cleared for training.** The full 417-query dataset is complete, but the current controller design is not leakage-safe and the configured stratified 80/10/10 split fails with the observed label counts. No LightGBM model was trained.

## Dataset and labels

- Rows: 417; unique non-empty `query_id`: 417 (one row per query).
- Each query has complete repeated latency for budgets 400/512/768/1024 (five runs per query-budget pair).
- `oracle_budget_095` and `oracle_budget_098` have identical distributions:

| Oracle label | Count | Fraction |
|---:|---:|---:|
| 400 | 259 | 62.1% |
| 512 | 113 | 27.1% |
| 768 | 39 | 9.4% |
| 1024 | 6 | 1.4% |

The 1024 class is extremely sparse; performance for this class will be unstable even under a corrected split.

## Feature availability and leakage

The current `train_budget_controller.py` excludes labels, latency columns, and `recall_at_*` columns. However, it includes all remaining columns, including features suffixed with 400, 512, 768, and 1024.

| Feature family | Available time | Can be used for a pre-retrieval single-shot budget choice? |
|---|---|---|
| `query_length` | Before retrieval | Yes |
| `entity_count` | Intended before retrieval, but currently null for all 417 rows | No useful signal in this dataset |
| `*_400` search-state features | After the 400-budget retrieval stage | No; only available to a controller deciding whether to continue after 400 |
| `*_512` search-state features | After the 512-budget retrieval stage | No; only available after reaching 512 |
| `*_768` search-state features | After the 768-budget retrieval stage | No; only available after reaching 768 |
| `*_1024` search-state features | After the 1024-budget retrieval stage | No; post-decision and too late to select the budget |
| `recall_at_100_*` | After retrieval/evaluation | Correctly excluded from model inputs; label/evaluation only |
| `latency_*` | After running each budget | Correctly excluded from model inputs; label/evaluation only |
| `oracle_budget_*` | Derived using all budgets' quality and repeated latency | Correctly excluded from model inputs; target only |

Therefore the current classifier would learn from future-stage search features while being evaluated as if it chose a budget before retrieval. This is temporal feature leakage and the evaluation would not represent an implementable controller. The feature availability markdown is too permissive: “before final exact retrieval” is not sufficient; availability must be tied to the exact decision point.

Additional data-quality findings: `entity_count`, `candidate_generation_failure_*`, and `truncation_risk_*` are null for all rows; these columns provide no training information. `score_gain_*` values should also be checked for consistent semantics across budget stages before modeling.

## Split audit

- The source data has exactly one row per unique `query_id`, so a row split would be query-disjoint if performed successfully.
- The current training code attempts a nested stratified split: 80% train, then split the 20% remainder equally into validation/test.
- This split fails: after the first split, the 1024 class has only one example in the held-out remainder, so the second stratified split raises `ValueError` (“least populated classes ... only 1 member”).
- Consequently, no valid train/validation/test sets or pairwise overlap counts are produced by the current code. The evaluation script attempts to reconstruct the same nested split and will fail for the same reason.
- Split IDs are not saved to a manifest; even after correcting the strategy, save explicit query ID lists and have evaluation load the exact test IDs.

## Evaluation audit

`evaluate_learned_controller.py` does compare fixed400/512/768/1024, oracle-adaptive, and learned-adaptive using test-set Recall@100 and median latency. It also reports savings and quality loss versus fixed1024. The high-level metric choice is appropriate for an offline comparison, but the current evaluation cannot run because the split fails, and the learned policy's feature set is unavailable at its implied decision time.

## Required changes before training

1. Define the controller as either (a) a pre-retrieval predictor using only pre-retrieval features, or (b) a sequential stop/continue policy using only features available after each completed stage. Do not feed 512/768/1024 state to a decision made before retrieval.
2. Decide how to handle the six 1024 labels. Prefer gathering more representative queries or use a prespecified split/CV design that explicitly reports the sparse class; do not silently remove or merge it after seeing test results.
3. Save train/validation/test `query_id` manifests, confirm pairwise disjointness, and make evaluation consume the saved test manifest.
4. Re-run leakage and split audits against the finalized feature list and policy; only then train LightGBM and evaluate once on the untouched test queries.

