# PLAID incremental runtime PoC — GPU results

Parity gate passed before this run. This is a 45-query, three-repetition diagnostic on existing indexes, not a full benchmark sweep.
All scenarios include query encoding and candidate generation in end-to-end latency. `Restart400+512+1024` is the sum of three complete independent searches; incremental shares one encoding and candidate-generation pass.
Qrels are used only for post-hoc quality metrics. No index was built.

- GPU: NVIDIA RTX A6000
- Torch: 1.12.1
- Cohorts: an earlier 9-query subset (3 hard / 6 easy) and an independent 36-query cohort (9 hard / 27 easy); three repetitions per query on an NVIDIA RTX A6000. Existing indexes were reused; private filesystem paths are omitted.

## Aggregated results

See `plaid_incremental_stage_savings.csv` for aggregate mean/p50/p95 and quality. Raw per-query/repetition stage timings are intentionally excluded from this public archive.

### Interpretation cautions

Incremental exact MaxSim scores are cached as scalars by PID; decompressed vectors are not retained. Runtime includes Python cache bookkeeping, budget selection, and per-stage CUDA synchronization.
Restart sums three complete searches and therefore repeats encoding and candidate generation. A separate per-query shared-encoding restart proxy can be derived by subtracting the duplicated encoding/retrieval portions.
The GPU legacy `score_pids` diagnostic counter is initialized only in-process for the oracle; default source behavior is not changed.
