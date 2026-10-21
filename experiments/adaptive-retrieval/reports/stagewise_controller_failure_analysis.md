# Stagewise controller: failure analysis and online feature audit

Date: 2026-10-16

## 1. Why the validation controller missed continuations

The oracle labels imply these stop/continue counts in the 42-query validation split:

| Decision point | Continue target | Validation positives | Predicted continue | Recall |
|---|---|---:|---:|---:|
| after 400 | oracle budget > 400 | 16 | 6 | 0.375 |
| after 512 | oracle budget > 512 | 5 | 0 | 0.000 |
| after 768 | oracle budget > 768 | 1 | 0 | 0.000 |

At 512, the validation model predicts all examples as “stop”, missing all five examples whose oracle budget is 768 or 1024. At 768, it again predicts “stop” for the one positive example. The 768 model has only four positive training examples, so its perfect training score is not evidence of generalization; it is a sparse-label/overfitting warning.

On the 42-query test set, 400 misses 6 of 16 examples that should continue; 512 misses 1 of 5; and 768 misses the sole 1024-oracle example. The overall learned policy therefore never selects 1024 in this test set. Its predicted budget distribution is 400:18, 512:14, 768:10, 1024:0.

Likely causes supported by the audit:

1. **Severe stagewise imbalance:** the 512 decision has only 45 positive cases over all 417 queries (35 train, 5 validation, 5 test); the 768 decision has only 6 (4/1/1).
2. **Weak generalization of the available signals:** the 400 validation continue recall is 37.5%, despite 16 validation positives. The 512/768 validation recall is zero. This suggests the current state features do not reliably rank future oracle continuation needs on this split; accuracy alone is misleading because “stop” dominates later stages.
3. **Small validation support:** a single 768-stage positive makes that stage’s validation score effectively one binary outcome, not a stable estimate.
4. **Threshold not calibrated:** the current rule is a fixed predicted-probability threshold of 0.5. No threshold was selected on a sufficiently large validation sample, and test data was not used to tune it.
5. **The target is an offline oracle choice:** `oracle_budget_098` is selected with all budgets’ measured recall and latency. The stagewise target `oracle_budget > stage` is a supervised proxy for that hindsight choice, not a direct prediction of marginal quality gain minus marginal runtime cost.

Do not respond to these misses by tuning against the 42-query test set. Threshold or feature decisions must use a larger, predeclared validation design, or the model must be re-evaluated with nested/grouped cross-validation before fixing a policy.

## 2. Are the features available at an online decision boundary?

Code inspection confirms the progressive API can expose the core approximate-search signals at a checkpoint:

- `server_two_stage_runtime.run_two_stage()` encodes the query and generates candidates once, then calls `approximate_search_stage()` successively with `previous_state=state` for 400 → 512 → 768 → 1024.
- Each call returns `state["signals"]` before `exact_refine_stage()` is called. The current `run_hybrid_adaptive()` already demonstrates a synchronous decision after each stage, using only the current checkpoint signals.
- The signal implementation computes cutoff margin, entropy, tail slope, score gain, and newly admitted candidate fraction inside `approximate_search_stage()`. With `PLAID_DEVICE=cpu`, tensors and signal reductions are CPU-side; the policy can read them before deciding to continue.
- In controller training, stage 400 features are allowed at the 400 checkpoint; stage 512 features are first allowed after reaching 512; stage 768 features are first allowed after reaching 768. The saved feature manifests show no future-stage columns.
- Query length is available before retrieval. `candidate_generation_failure` and `truncation_risk` are derived from qrels/final results or are empty in the dataset, and were dropped; they must not enter a real controller.

There is a previous full 417-query CPU progressive run (`results_cpu_lotte417/`), whose emitted signal snapshots cover all 417 queries for the run’s recorded stages. The measured `approximate_decision_state_ms` was:

| Recorded target | Queries | Mean signal-state time (ms) | Median (ms) | P95 (ms) |
|---:|---:|---:|---:|---:|
| 400 | 417 | 0.374 | 0.371 | 0.436 |
| 512 | 417 | 0.748 | 0.729 | 0.863 |
| 1024 | 417 | 1.106 | 1.095 | 1.286 |

That artifact’s metadata says CPU and one repetition, but it records targets 400/512/1024, not a separately timed 768 checkpoint. The 768 signals do exist in the full budget profile and the runtime loop supports that checkpoint, but a 768-specific signal-state latency distribution has not yet been measured in this saved runtime artifact.

### Important distinction

**Feature computability is validated structurally; learned-policy online operation and end-to-end latency are not yet validated.** The offline evaluator replays the learned decisions from saved per-budget feature rows and uses the repeated fixed-budget latency for the selected budget, then adds measured predictor-call time. It does not run a single live search and branch at each checkpoint. Therefore it does not capture the complete adaptive execution path, the cumulative stage cost, or prove that the final serialized model is integrated into the production retrieval loop.

The next proper validation is a CPU online replay harness that, for each held-out query, performs one candidate generation, expands the same progressive `previous_state`, invokes the trained stage model at the real checkpoint, finalizes exactly once, and logs each decision, feature-computation time, model-inference time, cumulative latency, and final Recall@100. Use the existing fixed and oracle policies as comparators on the identical query set. Do not collect extra queries until this checkpoint-level harness and its logs are verified; after that, expand the query cohort with a prespecified sampling plan that captures more rare high-budget cases.

