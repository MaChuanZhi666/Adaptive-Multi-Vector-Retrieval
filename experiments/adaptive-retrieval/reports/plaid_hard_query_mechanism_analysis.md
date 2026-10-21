# PLAID hard-query mechanism analysis

## Direct observations

- Old48 has 3 fixed400-sensitive queries (31, 40, 42). Independent94 has 9 (65, 83, 88, 100, 106, 112, 117, 128, 142).
- For every independent query, baseline and fixed400 have exactly the same pre-ndocs candidate count. The two strategies differ only at the ndocs-controlled filtering/refinement budget.
- Fixed1024 restores baseline Recall@100 for all nine hard queries. Therefore the 1024 path is capable of recovering the lost relevant documents.
- The frozen rule found 3/9 hard queries and missed 6; its independent precision was 14.29% with 18 false positives.

## Failure stage

All nine independent hard queries are classified as **B: truncation/pruning failure (mechanism-inferred)**. Candidate-generation failure is ruled out for the relevant documents visible in baseline because candidate generation is identical and occurs before ndocs is applied. A retained relevant document scored by the same exact scorer against a subset of competitors cannot acquire a worse rank solely from reducing the competitor set, so ranking/refine failure is not supported. However, per-PID candidate/prune traces were not saved, so this is a structural inference rather than direct PID-level observation.

## Feature stability

- Cross-feature separability-rank correlation between datasets: **-1.000**.
- Direction-consistent common features ranked highest: **centroid_top1_top2_gap_mean, candidate_density_per_token**.
- The old mean centroid-gap and embedding-norm thresholds do not transfer: the norm threshold catches 0/9 new hard queries; the gap threshold catches 3/9.
- Token-level worst-gap/quantile features, candidate cutoff margin, token candidate overlap, and independent prune-reduction ratio were not recorded. Their stability cannot be evaluated offline.

## Cross-dataset rules

- The frozen old two-feature OR rule on independent94: TP=3, FN=6, FP=18, hard recall=0.333, precision=0.143.
- Single-feature thresholds were fit in each source and applied only to the other source. These are diagnostic transfer tests, not proposed final rules. Results are in `plaid_cross_dataset_rule_evaluation.csv`.
- No new 2–3 feature rule is promoted from this offline analysis because the most mechanism-relevant requested inputs (worst-token gap and cutoff margin) are absent. Fitting a replacement only from aggregate means on independent94 would repeat the same overfitting mistake.

## Missing evidence

The current files do not contain per-token centroid scores/gaps, per-token centroid ranks, per-token candidate lists, candidate-score arrays around the cutoff, or per-PID membership before/after pruning. Remapped positive PIDs were also not included in the downloaded result package, so exact relevant-document ranks cannot be reconstructed. Placeholder figures explicitly mark these gaps rather than displaying invented values.

## Answers

1. **Mechanism:** ndocs=400 loses Recall@100 at the budget-controlled filtering/truncation step after common candidate generation.
2. **Stage for the nine queries:** all nine are most consistent with truncation/pruning; this is inferred from control flow and result monotonicity, not directly PID-traced.
3. **More stable feature:** none of the currently shared aggregate features is strong enough to replace gap mean confidently. Candidate density and candidate count can be compared cross-dataset, but their transfer metrics must be treated as exploratory.
4. **Feature versus threshold:** both are implicated. The direction of some aggregate signals is weakly consistent, but separability changes and the frozen thresholds fail badly. This is more than a simple threshold shift for embedding norm.
5. **Continue adaptive PLAID?** Yes as a mechanism study, not yet as a validated policy. The achievable latency/quality trade-off remains real, while detection is the unresolved component.
6. **Next features:** record **worst-token/lower-quantile centroid gap** and **candidate score margin at the ndocs=400 cutoff**. These directly represent ambiguous query tokens and truncation risk. Candidate overlap/density is the next supporting feature.
