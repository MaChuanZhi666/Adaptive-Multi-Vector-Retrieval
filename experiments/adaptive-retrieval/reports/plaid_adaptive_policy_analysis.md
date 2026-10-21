# PLAID simple adaptive ndocs offline analysis

## Scope

- Existing 13-config sweep only: 48 queries, 3 repetitions, LoTTE lifestyle/dev, 50,000 passages, TopK=100.
- No retrieval, indexing or model execution was repeated.
- Predictor inputs are restricted to query-encoding, centroid/probe and early-candidate statistics. Final latency, qrels metrics, final TopK results and minimum sufficient budget are excluded.
- Main classifier estimate uses strict outer leave-one-query-out validation. Each fold selects its feature, direction and threshold from the other 47 queries.

## Difficulty separability

- Best descriptive feature by oriented ROC AUC: `centroid_top1_top2_gap_mean` (AUC=0.913, Mann-Whitney p=0.0004).
- Nested LOOCV selected-feature frequencies: `{'centroid_top1_top2_gap_mean': 47, 'centroid_score_entropy': 1}`.
- Single-threshold LOOCV: hard recall=0.667, precision=0.235, FN=2, FP=13, accuracy=0.688.
- Two-variable OR LOOCV: hard recall=0.667, precision=0.444, FN=2, FP=5, accuracy=0.854.
- Auxiliary logistic LOOCV: hard recall=0.500, precision=0.500, FN=3, FP=3, accuracy=0.875.
- Nested LOOCV two-feature pair frequencies: `{('query_embedding_norm_std', 'centroid_top1_top2_gap_mean'): 47, ('query_embedding_norm_std', 'centroid_top2_mean'): 1}`.
- A rule fitted and evaluated on all 48 queries can reach 100% hard recall with feature `centroid_top1_top2_gap_mean` <= 0.111397, but its precision is only 0.333 with 12 false positives. This is resubstitution evidence, not held-out performance.
- Because only six queries are labeled hard, threshold and feature stability remain uncertain. The full-data rule in `plaid_candidate_online_rule.json` is descriptive and must not be reported as independent validation.

## Important label-policy alignment check

The six labels were defined by sensitivity to `joint_vlow`, which changes ncells, threshold and ndocs together. The new policy changes only ndocs.

- Queries with any ndocs=400 quality drop relative to baseline: [31, 40, 42].
- Overlap with the six joint_vlow-sensitive labels: [40, 42].
- ndocs=400-sensitive queries absent from the six-label target: [31].
- Queries escalated by the selected LOOCV rule: [25, 31, 32, 34, 40, 42, 43, 46, 47].
- Against the ndocs-specific target, the selected rule catches all 3 affected queries; its escalation precision is 0.333.

This mismatch limits what the classifier can recover. It is not valid to treat the six joint-vlow labels as a perfect ground truth for an ndocs-only controller.

## Offline policy replay

| Policy | Mean ms | P50 ms | P95 ms | QPS | Success@10 | MRR@10 | Recall@10 | Recall@100 | Hard recall | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed baseline | 160.123 | 158.725 | 174.095 | 6.245 | 0.9375 | 0.7776 | 0.7181 | 0.8704 | 1.000 | 0 | 42 |
| Fixed ndocs=400 | 133.064 | 131.653 | 141.880 | 7.515 | 0.9375 | 0.7776 | 0.7181 | 0.8501 | 0.000 | 6 | 0 |
| Adaptive 400/512 LOOCV | 133.762 | 132.623 | 142.638 | 7.476 | 0.9375 | 0.7776 | 0.7181 | 0.8635 | 0.667 | 2 | 5 |
| Adaptive 400/1024 LOOCV | 138.882 | 132.664 | 170.404 | 7.200 | 0.9375 | 0.7776 | 0.7181 | 0.8704 | 0.667 | 2 | 5 |
| qrels-strict Oracle | 130.466 | 129.518 | 136.201 | 7.665 | 0.9375 | 0.7793 | 0.7181 | 0.8704 | n.a. | n.a. | n.a. |

The adaptive replay uses the better recall-prioritized simple rule selected from the two cross-validated families: **two-variable OR**.

## Answers

1. **Do early features separate the six hard queries?** The best feature shows measurable distribution shift, but the small hard set and LOOCV errors mean separation is not robust enough to call clear or production-ready.
2. **Best cheap feature:** `centroid_top1_top2_gap_mean` is the strongest descriptive single feature in this sample. The nested-fold selections in `plaid_difficulty_rule_cv.csv` show whether that choice is stable.
3. **Can a simple rule identify the hard queries?** The honest estimates are the single- and two-variable LOOCV rows above, not the resubstitution rule trained on all 48 queries.
4. **Cost of prioritizing hard recall:** the selected two-variable OR policy produces 5 false positives and obtains 66.7% held-out hard recall; its precision is 44.4%. The apparent 100%/33.3% recall/precision rule is training-set-only and did not survive strict LOOCV. Guaranteed 100% protection with the current evidence therefore requires the degenerate fallback of escalating every query, which has 12.5% precision and removes the latency benefit.
5. **Quality recovered over fixed ndocs=400:** adaptive 400/512 recovers 65.9% of the Recall@100 gap; adaptive 400/1024 recovers 100.0%. Top-10 aggregate metrics were already unchanged at fixed ndocs=400.
6. **Latency retained versus baseline:** adaptive 400/512 improves mean latency by 16.46%; adaptive 400/1024 by 13.27%.
7. **Distance to Oracle:** Oracle mean is 130.466 ms/query. The adaptive policies are 3.296 ms and 8.416 ms slower, respectively.
8. **Design decision:** the low-by-default design is promising enough for a small independent online validation: adaptive 400/1024 restores the aggregate baseline quality while retaining 13.27% mean-latency improvement. It is not ready for deployment because there are only three ndocs-specific failure cases and the original six-label target is misaligned. Before the next test, relabel difficulty specifically as ndocs=400 quality loss, freeze the two-variable rule, and evaluate it without retuning on an independent query set.
