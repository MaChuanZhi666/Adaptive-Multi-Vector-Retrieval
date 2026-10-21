# PLAID two-stage adaptive policy failure analysis

## Scope and interpretation

Offline analysis only; no code changes and no benchmark reruns. Inputs were `plaid_two_stage_runtime.csv`, `plaid_two_stage_policy_results.csv`, and `plaid_approx_signal_stability.csv` (with the query labels and feature values carried in the existing runtime/feature records). The old set has 9 queries and the independent set has 36; runtime rows contain three repetitions per query.

The selected adaptive rule was trained/frozen on old queries using `approx_cutoff_margin`: at 400, high margin signals risk; at 512, low margin signals continued risk. The independent query results are therefore the relevant transfer test. Counts below are query counts, not repeated-run counts.

## 1. Query-level budget choices

| Dataset | Query group | stop@400 | stop@512 | continue@1024 |
|---|---|---:|---:|---:|
| old | easy (6) | 4 | 1 | 1 |
| old | hard (3) | 0 | 1 | 2 |
| independent | easy (27) | 7 | 10 | 10 |
| independent | hard (9) | 3 | 4 | 2 |

“Fixed400-sensitive” means Recall@100 at 400 is lower than Recall@100 at 1024. In independent, 9/36 queries meet this definition. Seven are assigned a budget below 1024: 83, 88, 100, 106, 112, 128, and 142. But only three of those choices actually reduce Recall@100 versus 1024: **100, 128, and 142**, all stopped at 400. Query 83, 88, and 112 are sensitive at 400 but 512 already recovers their 1024 Recall@100. Query 106 remains at Recall@100=0.333 at both 512 and 1024, so the larger budget does not repair its missing-candidate component.

Thus the actionable under-escalations are independent queries **100, 128, 142**. Their per-query Recall@100 is respectively 0.800→1.000, 0.333→0.667, and 0.750→1.000 (adaptive budget→1024). The adaptive policy sends 20 independent queries beyond 400 even though they are not fixed400-sensitive: **50, 56, 66, 67, 75, 81, 82, 84, 87, 90, 104, 111, 113, 116, 125, 136, 140, 143, 144, 145**. These are false escalations relative to the observed Recall@100 outcome; some also have candidate-generation misses, which cannot be remedied by this escalation.

On old, query 31 is fixed400-sensitive but stops at 512; 512 and 1024 have the same Recall@100 (0.714), so there is no observed Recall loss. Easy queries 25 and 43 are escalated despite no Recall@100 gain at the larger budget. Old has only three hard queries, so this is a small-sample check, not evidence of robust calibration.

## 2. Where the quality loss comes from

The independent adaptive policy's Recall@100 is 0.886 versus 0.908 for Always1024, a mean gap of about **0.0218**. The query-level gap comes from the three truncation-sensitive queries stopped at 400: 100, 128, and 142. For all three, candidate-generation Recall ceiling is 1.0: the relevant candidates were available before truncation, so the observed adaptive-vs-1024 loss is a **budget/truncation decision error**, not candidate generation.

Candidate-generation failure is a separate residual mechanism. The stored label means at least one qrels-relevant PID is absent from the generated candidate set; this occurs on independent queries **82, 90, 106, 118, 143 (5/36)**. It does not mean every relevant PID is absent. Their mean candidate-generation Recall ceiling is 0.949. Increasing ndocs cannot retrieve the absent PID. Query 106 is mixed: its remaining 1024 Recall@100 is only 0.333, and the trace marks both candidate-generation and truncation risk; the candidate-generation component persists at every tested budget. None of the three adaptive-vs-1024 Recall-loss queries is one of the five candidate-generation-failure queries.

Therefore, for the *incremental loss versus Always1024*, the proximate cause is under-escalation on 100/128/142. Candidate-generation failure explains a distinct portion of the quality ceiling that even Always1024 cannot remove; it is not the cause of the measured 0.886-vs-0.908 gap.

## 3. Latency decomposition

Independent-set mean latency (ms/query):

| Component | Frozen adaptive | Always1024 | Difference |
|---|---:|---:|---:|
| Query encoding | 20.348 | 20.378 | -0.030 |
| Candidate generation | 1.000 | 1.004 | -0.004 |
| Approximate filtering/checkpoints | 7.141 | 0 | +7.141 |
| Approximate decision/state work | 1.108 | 0 | +1.108 |
| Final exact-refine wall interval | 2.852 | 7.181 | -4.329 |
| **End-to-end total** | **31.405** | **28.578** | **+2.826** |

The policy spends about **8.25 ms** on approximate checkpoints and decision/state work, saving about **4.33 ms** in final exact refinement. The net stage arithmetic suggests roughly +3.92 ms, while the measured end-to-end delta is +2.83 ms; component timers are not perfectly additive because of GPU event/synchronization and wall-timer boundaries. The conclusion is unchanged: approximate checkpoints cost more than the smaller final refine saves. Candidate generation and query encoding are essentially identical between policies and do not explain the adaptive overhead. Always1024 performs one candidate-generation path and one full refine; adaptive performs the checkpoint work plus one final refine.

Full1024's legacy `score_pids` timing is recorded as one refine interval without internal GPU-event decomposition. Therefore the comparison supports the approximate-stage overhead conclusion, but not a fine-grained claim about exactly which internal Full1024 kernel accounts for each millisecond.

## 4. Are the approximate signals stable?

Hard/easy is based on the existing truncation-risk label at the corresponding budget. Values below are query-level medians; direction/AUC are taken from the existing stability analysis.

| Budget / signal | Old hard vs easy | Independent hard vs easy | Transfer assessment |
|---|---|---|---|
| 400 cutoff margin | 0.015625 vs 0.000000; oriented AUC 0.806 | 0.015625 vs 0.007813; oriented AUC 0.527 | Same nominal direction, but independent separation is nearly chance. The old-fitted cutoff over-escalates. |
| 400 tail slope | 0.015106 vs 0.016922; oriented AUC 0.556 | 0.013542 vs 0.018234; oriented AUC 0.607 | Same direction (lower slope associated with risk), but weak discrimination in both sets. |
| 400 candidate overlap / growth | overlap is constant 0; new-admitted fraction constant 1.0 | same constants | No discrimination at the initial checkpoint; these are structurally uninformative there. |
| 400 approximate top-100 overlap vs exact refine | 0.870 vs 0.800; oriented AUC 0.750 | 0.890 vs 0.870; oriented AUC 0.630 | Weak-to-moderate separation only. More importantly, this feature uses the later exact result and is unavailable to a live stop decision. |
| 512 cutoff margin | 0.007813 vs 0.011719; oriented AUC 0.563 (only 1 risk query) | 0.000000 vs 0.015625; oriented AUC 0.777 (5 risk queries) | Same direction, promising independent ordering but too few old positives and not enough to validate the policy end-to-end. |
| 512 overlap / new-admitted fraction | overlap constant 1.0; growth constant 0.21875 | same constants | No query-level separation at this stage. |
| 512 tail slope | old has 1 risk query; oriented AUC 0.875 | oriented AUC 0.548 | Not stable in strength; tiny old positive count makes the old AUC fragile. |

The signal story is not “all signals are useless”: cutoff margin has some ordering, especially at 512, and tail slope has a consistent but weak direction. However, **old-set threshold calibration does not transfer well at 400**, and candidate-overlap/growth values are constant at the tested fixed checkpoints. AUC direction consistency alone does not establish a useful decision rule. At 512, the old-set risk count is just 1/9, so threshold fitting there is particularly unstable.

## 5. Why the policy misses quality and loses time

The failure is a combination, with distinct roles:

1. **Signal/transfer:** the 400 cutoff-margin ordering collapses from old AUC 0.806 to independent AUC 0.527. On independent, the frozen rule sends many easy queries upward and still leaves three quality-improving hard queries at 400.
2. **Policy calibration:** the old threshold is conservative for recall on only 9 queries, but its transfer is not conservative in a useful way: it escalates 26/36 independent queries beyond 400 (72.2%), while missing 100/128/142. False promotions 20/36 dominate the compute waste; only 9/36 are fixed400-sensitive.
3. **Approximate-stage cost:** the checkpoint path adds about 8.25 ms before finalization, more than the 4.33 ms saved by refining fewer candidates. This is the direct reason adaptive is slower than Always1024 despite smaller exact-refine work.
4. **Candidate generation:** 5/36 independent queries have at least one relevant PID missing before truncation. This limits achievable quality at any ndocs, but does not explain the measured adaptive-vs-Always1024 gap; the three lost-Recall queries have candidate ceiling 1.0.

## Recommendation

Among the proposed directions, prioritize **B. make the approximate stage cheaper**, while keeping **A. signal validation** as a prerequisite for any new policy. This experiment already shows that a signal that triggers too often cannot pay for itself when checkpoints cost more than the refine work they avoid. Do not add a complex model or tune thresholds on this same independent set. Candidate generation is a separate follow-up only if the research goal is to improve the residual quality ceiling (especially the five affected independent queries); it will not fix the current adaptive latency regression. The evidence does **not** justify abandoning adaptive ndocs in principle, but it does justify rejecting this particular frozen rule/runtime path as a demonstrated win.

### Direct answers

- **Why does adaptive fail?** The 400 signal transfers poorly and causes both false promotions and three missed useful promotions; the approximate checkpoint computation costs more than the final refine savings.
- **Main failure source?** For quality delta versus Always1024: policy underpromotion on 100/128/142. For latency delta: approximate-stage cost. Candidate generation is a separate irreducible residual mechanism, not the source of that delta.
- **Next most valuable step?** Reduce approximate-stage cost first; re-evaluate a simple signal on a separately held-out set before claiming adaptive benefit. Work on candidate generation separately for the quality ceiling.
