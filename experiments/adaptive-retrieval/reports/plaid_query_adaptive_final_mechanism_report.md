# PLAID query-adaptive retrieval: final mechanism report

## Executive conclusion

The evidence supports **query-dependent ndocs sensitivity**, but not a robust or cost-effective adaptive controller yet. Some relevant PIDs are generated and later removed by the budget; increasing ndocs can recover quality for those queries. A separate set has relevant PIDs missing before truncation, which ndocs cannot fix. Many queries show no measured Recall@100 improvement at higher budgets. The mechanism is real, but simple static and runtime signals have not transferred reliably enough, and the tested staged approximate work costs more than it saves.

No source code was changed and no benchmark was run for this synthesis. Results below remain tied to their respective cohorts and protocols; the different independent subsets (27-query mechanism cohort, 36-query two-stage runtime cohort, 94-query frozen-rule validation) are not pooled.

## 1. Where query difficulty comes from

### Mechanism cohort and query-level counts

The directly traceable mechanism cohort is old=9 (3 selected hard + 6 fixed-seed easy) and independent=27 (9 selected hard + 18 fixed-seed easy). Classification distinguishes measured quality recovery from PID-level evidence: a relevant PID can be pruned without changing Recall@100.

| Dataset / cohort | Truncation-responsive (pure + mixed) | Candidate-generation-limited (includes mixed) | Easy / no observed budget quality gain, candidate-complete | Unknown quality outcome |
|---|---:|---:|---:|---:|
| old, n=9 | 3 (33.3%) | 0 (0%) | 6 (66.7%) | 0 |
| independent, n=27 | 8 pure + 1 mixed (9/27 = 33.3%) | 3 total (3/27 = 11.1%; one overlaps) | 14 (51.9%) | 2 (7.4%) |

“Truncation-responsive” means fixed400 quality is lower than a higher-budget result; “mixed” means both truncation and candidate-generation failure are present. Thus the truncation and candidate-generation columns overlap by one independent query (106), and counts should not be added without subtracting that overlap. The candidate-generation column includes query 90, which has a trace-level PID removal but no measured Recall@100 improvement. “Easy” here is specifically no observed low-to-high-budget quality gain and no candidate-generation miss, not a claim that the query is intrinsically simple for every metric. The independent two-query unknown bucket comes from missing quality-drop values in the stored feature table; it is not silently counted as easy.

The independent mixed query is **106**. The other eight responsive independent queries are 65, 83, 88, 100, 112, 117, 128, and 142. Candidate-generation misses in this cohort occur on 90, 106, and 131. Query 90 has a generation miss and a traced PID removal but no measured Recall@100 gain; query 131 has a generation miss without observed truncation responsiveness. Thus candidate-generation and truncation labels can overlap; they are not mutually exclusive failure flags.

PID trace adds an important caveat: in old, relevant PID rows are observed to be removed at the ndocs=400 boundary in 6/9 queries, but only 3/9 have a measurable Recall@100 benefit at higher budgets. In the 27-query independent trace cohort, relevant PID removal is recorded in 12/27; 8/9 known hard queries show the direct candidate-present-then-removed pattern, with 106 mixed. Per-PID pruning evidence therefore must not be equated with query-level quality loss.

The later two-stage runtime cohort is larger (old=9, independent=36; 9 hard + 27 easy for independent): 3/9 old and 9/36 independent are fixed400-sensitive by Recall@100. Candidate-generation-failure flags occur on 0/9 old and 5/36 independent; query 106 overlaps truncation sensitivity. These run-level counts are useful for policy analysis but are not substituted into the 27-query PID-trace table above.

### A. Truncation-limited queries

- The relevant PID is in the candidate-generation output, but the ndocs pruning boundary removes it. This is directly visible in the native trace for most selected independent hard queries.
- Raising ndocs can restore the PID and sometimes Recall@100. In the larger independent runtime cohort, hard queries match their own 1024 Recall@100 at budgets 200/300/400: 0/9, at 512: 4/9, at 768: 8/9, and at 1024: 9/9.
- Not every removed relevant PID changes Recall@100: multiple lower-budget queries retain enough other relevant results to leave the metric unchanged. Thus PID loss is a mechanism label; quality drop is the user-visible consequence.

### B. Candidate-generation-limited queries

- At least one qrels-relevant PID is absent from the generated candidate set. Increasing ndocs cannot bring that PID back.
- On the independent mechanism cohort, 3/27 have such a miss; on the later 36-query runtime cohort, 5/36 do. These are **query-level “at least one missing relevant PID” flags**, not a statement that all relevant PIDs are missing.
- Query 106 is mixed: some quality responds to expansion from 400, but a generation miss persists. The residual error at high ndocs is not attributable to truncation alone.
- This failure path needs candidate-coverage work (for example, a separately validated candidate-generation expansion strategy), not just ndocs adaptation.

### C. Easy / budget-insensitive queries

In the trace cohort, 6/9 old queries and 14/27 independent queries have known outcomes with no observed quality gain from low to high ndocs and no generation miss. Independent queries 90 and 131 are not called easy even though their measured quality is budget-invariant, because their candidate generation is incomplete; two more independent queries have unknown stored quality outcomes.

In the two-stage runtime, the adaptive rule escalated many queries whose Recall@100 did not improve at 1024. This confirms that “budget-insensitive” is an important population, while the available early features do not reliably isolate it.

## 2. Summary of adaptive attempts

### A. Static query-side feature rules

Features included token centroid-gap summaries (gap_p10, gap_p25, worst-token and bottom-k gaps), candidate density and related query-side statistics. The lower-tail features are more mechanism-aligned than gap_mean: gap_mean's direction was unstable; gap_p10 kept a lower-hard direction in the small 9-old / 27-independent mechanism comparison (oriented AUC approximately 0.944 / 0.722). But broader validation weakened this separation: gap_p10 AUC was about 0.807 old48 versus 0.642 independent94, and a later frozen two-feature v2 rule had 0/9 hard-query recall on independent94 (FN=9, FP=11). The old threshold was trained from very few hard cases and did not transfer.

The first frozen simple adaptive replay on independent94 had 33.3% hard recall (3/9), precision 14.3%, FN=6, FP=18. It recovered about 52.7% of fixed400's Recall@100 loss but remained below fixed1024/baseline quality; added feature extraction made it slower than fixed400. The v2 frozen rule missed all nine hard cases; replay Recall@100 was 0.8951 versus 0.9172 baseline and mean latency was 152.275 ms versus 148.061 ms baseline. These are negative transfer results, not evidence for an adaptive win.

**Why it failed:** small source-set hard counts, distribution/feature shift, and thresholds that trade missed hard queries for many false promotions. Feature direction consistency alone did not imply a useful operating point.

### B. Runtime approximate signals

Dynamic-signal instrumentation compared approximate candidate/TopK changes across ndocs checkpoints. Candidate growth was quota-driven (approximately 0.333 for both hard and easy in the earlier 200→300→400 probe; AUC=0.5), so it did not signal difficulty. Top75 churn at 300→400 had modest independent separation (AUC about 0.652) but reversed on old controls (AUC about 0.167); a frozen cutoff caught all 9 independent hard queries only by promoting 24/27 easy ones. The native filter did not expose its internal cutoff scores, so exact cutoff margins/slopes were unavailable in that instrumentation and were not inferred from ranks.

Two-stage approximate checkpoints later exposed cheap signals such as cutoff margin, tail slope, candidate overlap, and new-candidate fraction. On old→independent transfer, the selected 400 cutoff-margin signal fell from oriented AUC 0.806 to 0.527. At 400, candidate overlap/new-admission values were constant by construction. At 512, overlap and growth were also constant; apparent overlap with the later exact Top100 is not a valid online feature because it depends on future refinement.

**Why it failed:** early observable signals were either quota-constant, weak, unstable in strength, or unavailable until after exact scoring. AUC and small hard/easy median differences did not translate to calibrated decisions.

### C. Two-stage approximate-then-exact architecture

Engineering feasibility was established: an opt-in approximate-only checkpoint can inspect state at 400/512 and perform one exact finalize after choosing the budget. The six-query parity gate passed 18/18; the optimized incremental path also passed exact Top100 parity for its six-query gate plus nine hard queries. This demonstrates correctness feasibility, not speedup.

The genuine two-stage runtime experiment used 45 frozen queries and three repetitions. On independent36, the frozen old-threshold policy promoted 26/36 (72.2%), reached Recall@100 0.886 versus 0.908 at Always1024, and took 31.405 ms versus 28.578 ms. It stopped three quality-sensitive queries (100, 128, 142) at 400; another four 400-sensitive cases stopped at 512, which was sufficient for their observed Recall@100. Twenty queries were promoted without a Recall@100 gain. The approximate filtering and decision/state work cost about 8.25 ms/query, while the smaller final refine saved about 4.33 ms/query; net latency was worse.

The separate incremental optimization PoC summed to approximately 224.33 ms for 400→512→1024 versus 81.27 ms for a single full1024 stage (about 2.76×) under that run's timing protocol. It removed some wrapper overhead and maintained exact parity, but still repeated expensive search/refinement work.

**Why it is feasible but not beneficial yet:** API and parity are solved for a prototype; the architecture still performs too much approximate work before finalization, and the decision signal both over-promotes and misses useful escalations. Correctness is necessary but not sufficient for an adaptive win.

## 3. Final research conclusions

### 1. Does PLAID have query-dependent budget requirements?

**Yes, at the outcome level.** In the larger two-stage runtime cohort, Recall@100 at 400 was lower than 1024 for 3/9 old and 9/36 independent queries. The hard independent queries also show a graded recovery curve as budget increases. Other queries do not gain measured Recall@100 from more ndocs. This establishes heterogeneity in required budget, not a controller that can identify it cheaply.

### 2. Where does quality loss mainly come from?

For the fixed400→higher-budget quality gap, the dominant directly observed mechanism is candidate truncation: relevant PIDs were generated and then pruned, and higher ndocs often restores them. Candidate-generation misses are a distinct residual ceiling (3/27 in the mechanism independent cohort; 5/36 in the larger runtime cohort), with query 106 mixed. Across the three queries where the frozen two-stage adaptive policy lost Recall@100 against Always1024 (100, 128, 142), candidate coverage was complete; their loss came from stopping at 400.

### 3. Why is simple adaptation hard to generalize?

There are few hard queries relative to easy ones; feature distributions and feature-label ordering shift between collections; some runtime signals are mechanically constant; and the best post-hoc stability feature may not be available before exact scoring. A recall-first threshold selected from 3 old hard queries either misses independent hard cases or escalates many easy cases. Feature overlap with a label is not enough—the signal must be available online, calibrated on truly separate data, and cheap enough to pay for itself.

### 4. What future direction is most promising?

Treat the two failure mechanisms separately:

1. **For truncation:** make approximate continuation/early termination substantially cheaper, with cached candidate order/state and exact parity. Only then can a runtime ndocs decision have a chance to save time.
2. **For candidate-generation misses:** improve candidate coverage through a separate candidate-generation study (e.g. controlled ncells/nprobe expansion). ndocs escalation alone cannot solve this.
3. **For policy evaluation:** pre-register a small interpretable signal and evaluate it on a genuinely held-out collection; include all checkpoint, decision, synchronization, and final-refine costs. Do not retune on the existing independent sets.

## 4. Research roadmap

### Completed

- Fixed-budget and baseline comparisons, including old and independent query cohorts.
- Independent validation of frozen static predictors; false-negative and false-positive costs quantified.
- Pruning instrumentation and relevant-PID stage traces separating candidate-generation absence from post-generation truncation.
- Static token-level lower-tail feature analysis and dynamic-signal feasibility analysis.
- Incremental-search engineering feasibility and parity PoCs; current measured staged paths remain slower than one full1024 call.
- Two-stage approximate checkpoint runtime and query-level failure diagnosis.

### Unresolved

- A cheap, reliable early signal that transfers across datasets and is available before exact refinement.
- Lower-cost incremental approximate filtering and continuation that avoids recomputing/synchronizing work.
- Better candidate coverage for relevant PIDs missing before ndocs truncation.
- Larger, pre-registered independent validation with enough hard queries to estimate recall-risk reliably.

### Recommended next steps

1. **Do not present adaptive PLAID as successful.** Present this as a mechanism study with a negative/inconclusive controller result.
2. Optimize the approximate checkpoint path only behind parity gates; measure whether it can become materially cheaper than the exact refinement it replaces.
3. Investigate candidate-generation coverage independently from ndocs, and report its residual ceiling separately.
4. After those changes, test a simple dynamic signal once on a new held-out set. Stop if approximate overhead still exceeds saved refine cost or if recall-first transfer requires promoting most easy queries.

## Evidence and source files

- `outputs/plaid_query_adaptive/plaid_pruning_mechanism_instrumentation.md`
- `outputs/plaid_query_adaptive/plaid_truncation_risk_analysis.md`
- `outputs/plaid_query_adaptive/v2_features/plaid_v2_failure_diagnosis.md`
- `outputs/plaid_query_adaptive/v2_features/plaid_pruning_risk_v2_final_analysis.md`
- `outputs/plaid_query_adaptive/plaid_independent_validation_analysis.md`
- `outputs/plaid_query_adaptive/dynamic_signal_v1/plaid_dynamic_signal_final_analysis.md`
- `plaid_incremental_poc/plaid_incremental_poc_final_report.md`
- `plaid_incremental_poc/incremental_optimization_poc_report.md`
- `plaid_incremental_poc/plaid_two_stage_adaptive_final_report.md`
- `plaid_incremental_poc/plaid_adaptive_failure_analysis.md`
