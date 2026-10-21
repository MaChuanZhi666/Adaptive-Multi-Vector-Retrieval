# PLAID two-stage adaptive search: interface and parity design

## Goal and boundary

This design separates budget expansion from exact document scoring:

1. **Approximate stage:** candidate filtering, centroid-score reductions, budget selections, and stop/continue statistics only.
2. **Exact stage:** once the controller selects a final budget, decode only that budget's final refine PIDs, run MaxSim/GEMM once, and produce Top100 once.

The split belongs in `IndexScorer.score_pids()` immediately before its “Rank final list of docs using full approximate embeddings” block: after the GPU two-step filter has produced final refine PIDs, but before `lookup_pids()` / residual decompression / `colbert_score_packed()`. The legacy method stays unchanged; the experimental API is opt-in.

## Proposed experimental API

```text
state = scorer.approximate_search_stage(config, Q, pids, centroid_scores, budget=400)
state = scorer.expand_approximate_stage(config, state, budget=512)   # optional
state = scorer.expand_approximate_stage(config, state, budget=1024)  # optional
scores, top_pids, state = scorer.exact_refine_stage(config, state)   # exactly once
```

`approximate_search_stage()` computes/caches stage-1 thresholded candidate scores once, chooses top `budget`, and computes/caches stage-2 full-centroid scores only for rows newly included in the stage-1 prefix. It chooses top `budget//4` final refine PIDs, but does not call `lookup_pids()`, `ResidualCodec.decompress()`, or `colbert_score_packed()`.

`expand_approximate_stage()` reuses query/candidate/centroid inputs and cached score arrays, admits only newly selected rows, updates the stage-2 top set, and emits approximate-only signals. It also must not call exact decode or MaxSim.

`exact_refine_stage()` accepts the final approximate state's refine PID tensor, makes one `lookup_pids()` call, one packed MaxSim scoring call, and one final rank/Top100 extraction. State records an exact-refine call counter and becomes terminal after this call to prevent accidental duplicate execution.

## Reusable state

- Immutable references: `Q`, raw candidate PID tensor, centroid score matrix, threshold mask, candidate count/device/dtype.
- Cached per-candidate stage-1 thresholded approximate score vector.
- Candidate-row-aligned stage-2 full-centroid score vector and validity mask. The expansion computes only missing rows.
- Current stage-1 ordering/selected candidate IDs, current stage-2 top IDs/scores, final approximate refine PIDs, and budget history.
- Decision-only summaries: cutoff score/margin, tail slope/variance, selected-set overlap, new-row fraction, entropy, and saturation. These must derive only from the current/past approximate state.

The state does **not** contain decoded vectors, token×document GEMM matrices, or qrels. Exact MaxSim scores are not available until finalization.

## Approximate signals available before exact scoring

From the stage-1 and stage-2 score vectors and their current top-k selections, the API can cheaply expose: (a) score at the k and k+1 boundary and normalized cutoff margin; (b) local score-tail slope/variance; (c) overlap and new-member fraction between successive approximate candidate/refine sets; (d) score-distribution entropy; (e) retained candidate saturation. No signal may inspect qrels, exact scores, future budget results, or relevant PIDs.

## Parity protocol (first execution only)

Use 3 easy + 3 hard queries, no index build. For each query and target budget in `{400,512,1024}`:

1. Build a fresh approximate state and advance only through budgets up to the target.
2. Assert exact-refine call count remains zero at every approximate checkpoint.
3. Finalize exactly once; compare final Top100 IDs and ordering, per-PID MaxSim scores, Recall@100, Recall@10, MRR@10, and Success@10 against unchanged `score_pids()` at that budget.
4. Any mismatch stops the work before runtime measurement.

Exact parity can be preserved because approximate filtering uses the same stage-1 thresholded reduction, same `torch.topk` budget semantics, same full-centroid reduction, and same `ndocs//4` selection as legacy GPU `score_pids`; final exact work uses the same decoder and `colbert_score_packed`. It is not guaranteed by design alone: ties/non-deterministic top-k ordering, dtype/device differences, or a mistaken prefix cache can alter selected PIDs. Therefore stage PID/score parity plus final exact-score/ordering parity is a mandatory gate.

## Later runtime experiment (not part of the initial gate)

Only after 6-query parity passes, run the predeclared 45-query sample and compare Full400/512/1024 with Two-stage stop@400, 400→512 stop@512, and 400→512→1024 then one exact finalize. Time boundaries must be synchronized and shared encode/retrieve counted once; report approximate, decision, final decode, MaxSim, TopK, synchronization and total latency. Include warm-up and repeated matched-order runs.

Static risk labels and qrels may be used offline for retrospective feature evaluation and oracle upper bound, never by the online decision rule. Candidate-generation failures (e.g. query 106's known failure component) remain residual: increasing `ndocs` cannot recover a relevant PID absent from candidate generation.

## Expected benefit and limitation

If the policy stops at 400, exact refinement covers about 100 rather than 256 documents (up to 61% fewer exact document rows); stop at 512 covers about 128 (up to 50% fewer). At 1024, exact document work is still the same 256 rows as Full1024; only the progressive approximate/decision overhead is additional. No latency gain is promised until measured. Approximate-only checkpoints create actual early-termination feasibility, but whether their signals predict truncation risk robustly across old/independent datasets remains an empirical question.
