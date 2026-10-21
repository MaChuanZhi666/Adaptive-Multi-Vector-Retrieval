# PLAID pruning-risk mechanism instrumentation

- Selected subsets: independent 9 hard + 18 fixed-seed easy; old 3 hard + 6 fixed-seed easy.
- Existing indexes were reused. No training, indexing, full sweep, threshold tuning, or classifier fitting was performed.

## Directly observed

- Per-query token centroid statistics and native selected-PID membership were recorded for 36 queries.
- The raw trace contains 447 rows across ndocs=400/512/1024; the 400-only derived classification is in `plaid_relevant_pid_trace_400_failure_types.csv`.
- At ndocs=400, 31 traced relevant-PID rows are absent after native filtering among 61 hard-label relevant-PID rows.
- In the independent hard subset, 8/9 queries have all traced relevant PIDs present before filtering but at least one removed at ndocs=400 (direct candidate-truncation evidence). One additional query is mixed: it has both pre-candidate misses and a budget removal.
- 26 corresponding higher-budget trace rows show the relevant PID retained at ndocs=512/1024.

## Cutoff limitation

- The native `filter_pids` extension returns selected PIDs and vector-count metadata but does not expose the approximate score array used internally. Therefore rank-399/400/401 scores, cutoff margins, and local score slopes are NA. They were not approximated. A tiny C++/pybind instrumentation change would be required for direct cutoff-score measurement.

## Interpretation

- A relevant PID present before filtering and absent after ndocs=400 is direct evidence of loss at the native candidate-truncation boundary; the trace cannot distinguish internal centroid pruning from the subsequent ndocs budget.
- The independent hard set is therefore mostly a truncation-risk mechanism, with query 106 showing a mixed candidate-generation/truncation pattern. Higher budgets retain many, but not all, traced relevant PIDs; final quality recovery must still be evaluated with the existing per-query benchmark results.
- Token-level worst/lower-quantile gap stability is reported separately for old and independent subsets. Direction consistency is the criterion for a cross-dataset signal; no online rule is proposed here.
- Candidate overlap and exact cutoff margins remain unknown because the native API does not expose per-token PID sets or approximate score arrays.

## Status

- Direct: token gaps, selected-PID membership, budget-specific trace, and approximate candidate rank when the PID is in the pre-filter list.
- Inferred: candidate truncation is the dominant mechanism for 8/9 independent hard queries; query 106 is mixed.
- Unknown: exact internal cutoff rank/score, per-token overlap, and whether a different threshold would be the causal trigger.