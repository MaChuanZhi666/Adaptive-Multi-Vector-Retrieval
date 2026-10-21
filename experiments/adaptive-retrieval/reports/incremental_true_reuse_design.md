# PLAID true incremental reuse: CUDA/kernel design (source audit only)

## Scope and verdict

This is a source-level design review of the isolated incremental PoC. No code was changed, no benchmark was run, and no index was rebuilt.

The current opt-in API already caches several *values* by candidate row, so its 224 ms cumulative latency is not evidence that the same PID's exact MaxSim or decompression is repeatedly executed. The remaining cost is principally that the same final work is split across budget-sized GPU/CPU launches and repeated selection/reduction; there is no persistent kernel-side search state. “True reuse” should first mean persistent GPU-resident buffers and append-only work on newly admitted rows, not caching every decoded document indefinitely.

## Current dataflow and repeated work

### 1. Filter / approximate-score stages

On the GPU path used by the PoC, filtering is implemented in Python/Torch in `source/baseline/ColBERT/colbert/search/index_storage.py::score_pids` and `score_pids_incremental`; `search/filter_pids.cpp` is loaded only for the CPU path. `CandidateGeneration.get_cells()` computes centroid-query scores once in `candidate_generation.py`, and the incremental state retains the resulting `centroid_scores` and candidate PID tensor by reference.

In `score_pids_incremental()`:

- Stage-1 thresholded per-candidate approximate scores are computed once at initialization and retained in `state["stage1_approx_scores"]`.
- Each budget reruns `torch.topk(stage1_approx_scores, k=ndocs)` over the full cached vector. It does not recompute each score, but does repeat the selection scan/work.
- Stage-2 full-centroid approximate scores are cached in `stage2_approx_scores` behind `stage2_valid`. On expansion, only previously unseen candidate rows in the new stage-1 prefix are scored. Already-valid PIDs are not rescored.
- Each budget reruns top-k over all stage-2 scores in its selected stage-1 prefix to choose `ndocs//4` refine PIDs. Existing scalar scores are reused, but the prefix is re-read and selection rerun.

The legacy CPU `filter_pids.cpp` behaves differently: its first pthread pass computes local heap scores and emits only selected PIDs; scores are discarded. It then invokes `filter_pids_helper()` again on those PIDs with all centroid codes enabled, recomputing the per-document approximate score. Therefore a CPU incremental implementation must explicitly return/cache `(PID, score)` from pass one. That C++ file is not the active GPU PoC path.

### 2. Decompression

GPU `lookup_pids()` goes through `ResidualEmbeddingsStrided.lookup_pids()` → `ResidualCodec.decompress()` → `decompress_residuals.cu`. The CUDA decompressor allocates a fresh output tensor and decodes every embedding token for each PID in the input batch. It has no cache or append offset.

The opt-in incremental API gates lookup on `maxsim_valid`, so a PID already scored in an earlier stage is not passed to decompression again. The resulting `D_packed` vectors are deleted after scoring and not retained. Thus, under a live state and nested refinement, decompression is not duplicated per PID, but each stage pays a new allocation/launch for its newly admitted PID batch. Full1024 decodes its final refine set in one batch; incremental decodes the same final set as batches of about 100, then 28, then 128 PIDs (actual counts depend on candidate count and set nesting).

### 3. GEMM / MaxSim

GPU `colbert_score_packed()` in `modeling/colbert.py` performs a dense `D_packed @ Q.T` with Torch/CUDA GEMM, then `StridedTensor(...).as_padded_tensor()` and `colbert_score_reduce()` reduce over document tokens and sum query-token maxima. `modeling/segmented_maxsim.cpp` is the CPU implementation; it is not the GPU kernel in this run.

The current state stores only each PID's final scalar MaxSim in `maxsim_scores`, guarded by `maxsim_valid`; already-scored PIDs are not sent to GEMM again. It does not preserve the token-by-document-token product matrix or per-query-token maxima. For fixed Q and fixed full document representation, a document's MaxSim is independent of other PIDs and can be finalized once. There is no useful cross-budget partial MaxSim to resume for an unchanged document: the natural reuse unit is the final scalar score. A token-level matrix cache would be much larger and provide no benefit unless the implementation deliberately computes documents in token chunks.

### 4. TopK

The optimized PoC does maintain an incremental exact-score Top100 for nested refine sets: prior Top100 plus newly scored PIDs are merged with `torch.topk`. If a prior PID drops from the current set or the merge detects a tie, it falls back to sorting the whole current refine set. The fallback and `.item()` checks cause synchronization. In contrast, the two approximate filter selections (stage-1 top-ndocs, stage-2 top-ndocs/4) still each rerun full-prefix `torch.topk` at every budget.

## Proposed persistent state

Keep all state on the same CUDA device and index it by a stable candidate-row ID (not by repeatedly materialized PID lists):

| Buffer | Owner/lifetime | Update rule |
|---|---|---|
| Q, candidate PID tensor, centroid score matrix, threshold mask | query state | immutable for query lifetime; created once by retrieve |
| stage1 score `[N]` + valid bit (or immutable after initial pass) | filter state | compute once for all raw candidate rows |
| stage2 score `[N]` + valid bit | filter state | append scores only for newly admitted stage-1 candidate rows |
| stage1 rank/order (optional) | filter state | compute once; use prefix for increasing budgets, with explicit deterministic tie policy matching legacy |
| current stage-2 top-k IDs/scores | filter state | merge prior top-k with newly scored candidate rows; maintain the top `ndocs//4` required for each budget |
| packed document arena + per-candidate offset/length + valid bit (optional) | query state or bounded scratch | write decoded embeddings for new refine PIDs only; reuse only while memory budget allows |
| exact MaxSim scalar `[N]` + valid bit | score state | compute once for newly refined rows; never retain token×token GEMM output |
| Top100 IDs/scores | result state | merge prior Top100 with new exact scores; keep tie handling deterministic and parity-compatible |

Two memory modes are recommended:

1. **Default bounded-memory mode:** do not retain decoded D vectors; current exact-score validity already prevents repeat decode for the same PID. Allocate a reusable packed scratch arena sized for the largest expected new batch to reduce allocator churn, then overwrite it after GEMM completes.
2. **Optional decode-cache mode:** retain decoded normalized FP16 vectors only for the query's admitted refine PIDs. This can save decompression only if budget sequences can revisit PIDs (restart, rollback, non-nested sets) or if downstream scoring is repeated. Under strictly nested 400→512→1024 and exact-score caching, it does not reduce decompression arithmetic; it only avoids a hypothetical later re-score, at potentially high GPU-memory cost.

## Minimal C++/CUDA modification points (design only)

1. **GPU filter state API:** add an opt-in extension entry point in a new `search/filter_pids_incremental.cu` (or a focused `.cu` companion), called only by `score_pids_incremental()`. Inputs: candidate row IDs/PIDs, cached stage-1 scores, centroid score matrix and validity buffers, previous selected/top-k IDs, target `ndocs`. It should compute stage-2 approximate scores only for new rows, then update top-k without copying scores to Python. Do not repurpose `filter_pids.cpp` for GPU: it is pthread/CPU code and returns only PIDs.
2. **Stage-2 reduction/top-k fusion:** fuse token centroid gather + MaxSim-like approximate reduction into one kernel for newly admitted rows, write scalar score into the persistent `[N]` array, and update a bounded top-k structure. Ensure reductions exactly match `colbert_score_reduce` semantics and retain a deterministic tie-break (e.g., score descending, candidate-row ID ascending) only if parity baseline adopts the same ordering; otherwise preserve the current legacy tie behavior via a fallback.
3. **Incremental CUDA decompressor:** extend the GPU residual-decompression wrapper in `indexing/codecs/residual.py` and kernel `indexing/codecs/decompress_residuals.cu` to accept selected embedding/PID rows plus precomputed output offsets and a caller-owned output arena. Kernel writes only new rows; it must not allocate/zero a full output tensor for each stage. Keep existing public decompression behavior unchanged; use a separate opt-in entry point.
4. **GEMM/reduction:** do not initially write a custom MaxSim kernel. Continue using cuBLAS-backed GEMM for new D rows, followed by current reduction. Batch all newly admitted rows per continuation into one GEMM. Only investigate a fused GEMM+segmented-max reduction if profiler shows material intermediate matrix traffic or launch overhead; protect score parity (FP16 accumulation/order may shift scores).
5. **Top100 update:** move merge and tie/cutoff logic into CUDA/Torch tensor ops with no `.item()` synchronization in the continuation hot path. One top-k over at most prior-100 plus newly scored rows is the intended cost. On initial stage, compute exact Top100 from initial exact scores. Store compact indices and scores, not a full sorted order of all refined PIDs.

## Expected reduction in work (not a speedup promise)

For a 1024 target, `ndocs//4` means at most 256 exact-refine documents. With nested refine sets, the current schedule processes approximately 100 + 28 + 128 = 256 distinct PIDs. A one-shot full1024 processes the same 256 PIDs. Therefore:

- **Exact document decompression and MaxSim arithmetic at final budget:** theoretical reduction is **0%** if the adaptive path reaches 1024; current validity masks already prevent PID-level repeats.
- **Stage-2 approximate scoring:** up to 1024 distinct stage-1-selected rows are needed by the final 1024 budget; current cache already limits it to newly selected rows, so arithmetic reduction vs one-shot full1024 is also approximately **0%**. The opportunity is batch/fusion and avoiding repeated prefix top-k, not eliminating score rows.
- **If early stop occurs at 400:** only about 100 of 256 final-refine PIDs require decompression/MaxSim: up to **61% fewer** exact-document rows than full1024. At 512: 128 of 256, **50% fewer**. This is the real work-saving case, contingent on acceptable quality.
- **If continuing through all budgets:** kernel launch/batch overhead plus repeated stage selection/TopK is added to roughly the same total arithmetic; this explains why persistent buffers alone cannot guarantee a 224 ms → 81 ms improvement. To approach full1024, the controller must avoid materializing exact intermediate results at every checkpoint (use cheap stage signals), or use one-shot 1024 scoring after a decision.

No defensible millisecond savings can be calculated from current measurements because there is no CUDA-event breakdown per kernel and the current test is one pass. Source-derived row counts above are the appropriate estimate until an instrumented, parity-gated microprofile is run.

## Recommended implementation order

1. First instrument current path with CUDA events around stage-1 scoring, stage-1 selection, new stage-2 score rows, stage-2 selection, decode, GEMM, reduction, and TopK; separately log row/token counts and cache hits. Do not run a benchmark sweep.
2. Implement GPU-resident stage-2 score + top-k update for only new rows; parity-test on existing 6+9 set.
3. Add a reusable packed decode scratch buffer and new-rows-only decompressor API only if CUDA-event data shows allocation/decompression launch cost matters.
4. Preserve exact Top100 and per-stage score parity against unchanged `score_pids()` at 400/512/1024 before any timing claim.

## Direct answers

1. **Are 400-stage scores saved?** Yes. Stage-1 approximate scores and per-candidate stage-2 approximate scores are retained in the experimental Python/Torch state. Already-valid rows are not rescored. The full-prefix top-k selection itself repeats per budget.
2. **Is decompressed data cached?** No. Decoded vectors are transient and discarded, but exact-score validity prevents re-decoding the same PID in a nested live continuation. New PIDs alone are decompressed at each extension. The CUDA kernel allocates a fresh output per call.
3. **Can MaxSim be extended from a partial token-document score matrix?** No such matrix or partial maxima are currently saved. They are unnecessary for fixed-query, full-document MaxSim when each PID is scored once; save its final scalar. New documents require independent GEMM/reduction.
4. **Is incremental TopK present?** Exact-score Top100 merge is present in the PoC, with full-sort fallback on non-nested sets/ties. Approximate filter top-k stages still rescan their current score prefixes each budget, and tie checks may synchronize.
5. **What should be changed first?** Make stage-2 score generation and its top-k update a single append-only GPU operation; then only if profiling justifies it, reuse a packed decode output arena. Do not cache token-level MaxSim matrices by default.
