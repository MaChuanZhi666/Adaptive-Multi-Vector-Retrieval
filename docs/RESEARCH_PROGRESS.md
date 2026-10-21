# Research Progress

For the full history and curated archive of prior experiments, see [Research Archive](RESEARCH_ARCHIVE.md).

- **Status snapshot:** 2026-10-21
- **Project:** Query-adaptive computation for PLAID/ColBERT-style multi-vector retrieval, with RAG as the motivating application.

## Research question

Can a retrieval system spend less computation on queries that are already well served by a smaller search budget, while preserving retrieval quality for queries that need deeper search? The current experimental variable is the candidate/search budget (`400`, `512`, `768`, or `1024`). Extending adaptation to other PLAID stages, including pruning, is a future hypothesis—not an evaluated result.

## Completed work

- Established a CPU execution path for the retrieval experiments to avoid GPU-to-CPU transfer effects in latency measurements. Existing PLAID scoring and candidate-generation behavior have not been changed for the profiling study.
- Built a multi-budget profiling and analysis workflow, including a JSONL profile format, feature extraction, checkpoint/resume support, oracle-budget analysis, and quality/latency plots.
- Completed real four-budget profiling for **417 LOTTE lifestyle-dev queries**. Recorded retrieval metrics, latency, and search-state signals including cutoff margin, tail slope, score entropy/gain, candidate overlap, and Top-K churn.
- Audited controller feature timing and identified that a stagewise controller must not use features from later budgets to decide whether to continue at an earlier budget. Final recall and oracle labels are labels/evaluation data, not valid controller inputs.
- Reviewed LAET, DARTH, and OMEGA as related work. Their stopping and multi-K ideas are useful inspirations, but they operate on different search systems; the current PLAID experiments are not reproductions of those papers.

## Preliminary results: 417-query profile

For the 417-query profile, the oracle budget is defined relative to the measured fixed-1024 result: a budget is eligible when `Recall@100(B) >= alpha * Recall@100(1024)`, and the lowest-latency eligible budget is selected. At both `alpha=0.95` and `alpha=0.98`, the observed distribution was:

| Oracle budget | Queries |
|---:|---:|
| 400 | 45 |
| 512 | 231 |
| 768 | 101 |
| 1024 | 40 |

This is evidence that the measured queries have different budget/quality trade-offs. It is **not** evidence that a deployable adaptive controller can identify the right budget, nor that it outperforms fixed-1024. The oracle sees outcomes from all budgets and is only an offline reference. Fixed-1024 is also not an exact-search ground truth.

Single-run mean latency in this profile was 43.16 ms (400), 35.98 ms (512), 38.02 ms (768), and 42.29 ms (1024). The non-monotonic values indicate measurement variability; repeated latency measurements are required before drawing performance conclusions.

In a separate earlier independent-query comparison, the observed results were:

| Method | Recall@100 | Latency (ms/query) |
|---|---:|---:|
| fixed400 | 0.8502 | 36.26 |
| fixed512 | 0.8754 | 37.55 |
| fixed1024 | 0.9080 | 41.71 |
| HybridAdaptive | 0.8793 | 60.40 |

That heuristic adaptive run did **not** outperform fixed-1024 in either recall or latency. These measurements are from a separate cohort and must not be conflated with the 417-query profile above.

## Ongoing work

A real CPU profile is in progress for **2,931 LOTTE pooled-dev queries** at the four budgets. At the latest server check (2026-10-21 09:11, server local time), the runner was active and its log had reached `query_id 1521`. This is a progress indicator, not a final coverage audit.

The runner currently rewrites the full JSONL checkpoint after each completed query. Consequently, the live file's line count can temporarily appear lower while a rewrite is underway, and repeated full-file writes add I/O overhead. Final coverage and JSON validity must be checked only after the run finishes. Raw query data, indexes, model files, and full experiment outputs are intentionally not included in this repository snapshot.

## Not yet established

- No learned budget controller has been trained and validated on an independent test set.
- No end-to-end online stagewise replay has yet established that decision features are available at the actual decision boundary, or that adaptive execution reduces total latency after decision/model overhead.
- No adaptive pruning experiment has been implemented or validated.
- No end-to-end RAG answer-quality, evidence-coverage, or groundedness evaluation has been completed.
- No full LAET, DARTH, or OMEGA reproduction has been completed on their original benchmarks.

Therefore, the current work does not support claims that learned adaptation, pruning adaptation, or RAG-level quality/latency improvements have been demonstrated.

## Next steps

1. Finish pooled-dev profiling and validate query IDs, all four budget records, required metrics, and feature coverage; preserve an immutable snapshot after validation.
2. Repeat latency measurements using the pre-locked training-query sampling plan, keeping the frozen test cohorts out of tuning.
3. Run CPU online replay with one candidate-generation pass, real stage boundaries, incremental state reuse, explicit controller overhead, and one exact finalization; compare final quality and end-to-end latency.
4. Train a simple controller only after the data and leakage audits pass, and evaluate it once on the untouched test split against fixed-budget and oracle references.
5. Pilot pruning as a separate stage only if its decision signals are observable online and its contribution can be isolated. Otherwise, keep it as future work.
6. Add RAG answer/evidence metrics only when a reproducible downstream RAG evaluation is available.

## Reproducibility and claim discipline

Keep raw datasets, indexes, private paths, credentials, and large generated result files outside this public repository unless their redistribution is explicitly permitted. Report per-query counts, measurement repetitions, confidence intervals or paired tests where applicable, and distinguish offline oracle bounds from deployable controller results. Do not claim an adaptive win unless the frozen-test, end-to-end comparison supports it.
