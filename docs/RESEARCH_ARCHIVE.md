# Research archive and project history

This archive collects the research work available in the project workspace as of 2026-10-21. It is broader than the latest progress snapshot: it includes earlier multi-vector retrieval benchmarks, adaptive-search investigations, runtime engineering, profiling/controller audits, and related-paper study.

## Research trajectory

### 1. Multi-vector retrieval benchmark and systems study

The initial work established a reproducible LoTTE lifestyle/dev comparison involving PLAID, WARP, and GIGP+/IGP-related implementations. Measurements covered GPU and CPU runs, retrieval quality, latency/QPS, index size/build time where available, GPU memory, and PLAID stage timing. The collected benchmark package reports 50,000 passages and 48 queries for its formal comparison. These are implementation- and hardware-specific measurements, not a universal ranking: PLAID/WARP GPU runs used a Tesla T4 while GIGP+ used an RTX A6000, and timing boundaries differ. See [`experiments/multivector-benchmarks/`](../experiments/multivector-benchmarks/).

A separate same-A6000 study compared PLAID and GIGP+ configurations on the same 48-query cohort. It records retrieval-only timings and explicitly distinguishes PLAID timing that includes query encoding from native GIGP+ timing using precomputed query embeddings. The measured recall-matched table did not establish a general GIGP+ speedup; use only directly measured configurations and preserve the timing-scope caveat.

### 2. Query-adaptive PLAID exploration

The research then moved from system comparison to the question of whether a fixed candidate budget (`ndocs`) treats all queries equally. Experiments explored static query features, runtime signals, hard-query protection, and adaptive budget policies. The mechanism analysis found several different cases: relevant PIDs may be present in candidate generation but removed by truncation; they may be absent before truncation; or a larger budget may not improve observed quality. Thus raising `ndocs` is not a universal repair.

Early small-cohort rules showed promise in selected analyses but did not transfer reliably to independent query cohorts. The consolidated failure analysis therefore treats query-dependent budget sensitivity as supported, while treating a reliable low-overhead controller as unsolved. The summary and failure analysis are in [`experiments/adaptive-retrieval/reports/`](../experiments/adaptive-retrieval/reports/).

### 3. Full-CPU fixed-budget and heuristic-adaptive comparison

The retrieval pipeline was run on CPU for candidate generation, approximate search, adaptive decision, and exact finalization. This removes GPU-to-CPU transfer from that experiment's measurement path; it does not imply CPU is faster than GPU. In an earlier independent-query comparison:

| Method | Recall@100 | Latency (ms/query) |
|---|---:|---:|
| fixed400 | 0.8502 | 36.26 |
| fixed512 | 0.8754 | 37.55 |
| fixed1024 | 0.9080 | 41.71 |
| HybridAdaptive | 0.8793 | 60.40 |

The heuristic adaptive run did not beat fixed1024 in recall or latency. This cohort is distinct from the later 417-query budget profile and should not be merged with it.

### 4. Incremental reuse and two-stage execution

An approximate-to-exact two-stage design was investigated, including parity checks, runtime decomposition, and true reuse of prior candidate/scoring state. The key engineering finding was that caching avoids repeating some work, but repeated stage setup, tensor assembly, top-k/sorting, synchronization, and incremental batches can outweigh those savings. On the measured GPU prototype, the incremental 400→512→1024 path preserved parity but was slower than one Full1024 pass. A design that is cheaper than restarting several independent searches is not automatically cheaper than a single full-budget search.

The archived reports distinguish work actually reused from work repeated and include follow-up optimization ideas. No speedup is claimed until remeasured end-to-end with parity retained.

### 5. Four-budget profiling and oracle analysis (417 queries)

A real CPU profiling run covered 417 LOTTE lifestyle-dev queries at budgets 400, 512, 768, and 1024. Each query was run at each budget; the JSONL profile records query text, retrieval metrics, latency, candidate information, and search-state features. Full per-query records are deliberately not included in this public archive.

An offline oracle was defined against each query's measured fixed1024 Recall@100: select the lowest-latency budget whose recall is at least `alpha` times the 1024 result. For both `alpha=0.95` and `alpha=0.98`, the reported distribution was 45 queries at 400, 231 at 512, 101 at 768, and 40 at 1024. This indicates heterogeneous observed quality/latency trade-offs in this cohort. It is an offline upper-bound analysis, not an implementable controller result; fixed1024 is not exact ground truth.

Single-run mean latency was non-monotonic (43.16, 35.98, 38.02, and 42.29 ms/query for 400/512/768/1024), so timing noise and experimental order need further control. Later repeated-measurement and online-replay work was started, but the independent test cohorts were kept frozen and must not be used for threshold tuning.

### 6. Learned-controller and online replay audit

Controller infrastructure and audits exposed important validity issues before any performance claim: stage features are available only after their stage; later-budget features cannot be used at an earlier decision; recall/oracle/answer-quality fields are labels or evaluation outcomes, not inputs; sparse high-budget labels make stratified splits fragile; and query IDs must be isolated and saved in an immutable split manifest.

The current archive does not claim a leakage-safe, independently validated learned controller or an end-to-end adaptive win. Online replay was treated as a prerequisite: one candidate-generation pass, same incremental state across real budget boundaries, model invocation at the actual decision point, one exact finalization, and explicit accounting of model and stage overhead. The status of a running pooled-dev job is intentionally not frozen into this archive; update the progress snapshot only after validating its final coverage and output integrity.

### 7. Related work and RAG motivation

LAET and DARTH motivate learned early termination and target-recall-aware search; OMEGA motivates multi-K search through reusable search states. Their original systems and benchmark settings differ from PLAID/ColBERT-style multi-vector retrieval. The work in this repository is a transfer of ideas and a study of compatibility, not a completed reproduction of those papers. Original benchmark data/code requirements were not met for a full reproduction.

RAG is the motivating downstream use case: retrieval efficiency matters when retrieval is a meaningful share of the overall request cost, while answer quality depends on evidence quality and downstream generation. No end-to-end RAG answer EM/F1, evidence coverage, or groundedness experiment has yet been established. Adaptation in PLAID pruning and other stages remains a research hypothesis, not a completed result.

## What the evidence currently supports

- The measured query cohorts exhibit query-dependent budget/quality sensitivity.
- Full-CPU execution is available for a controlled retrieval profile, and four-budget profiling infrastructure exists.
- Earlier heuristic adaptation and an incremental GPU prototype did not establish an end-to-end improvement over a single fixed high-budget pass.
- The 417-query oracle distribution is useful for motivating adaptive allocation, but cannot be presented as deployable adaptive performance.
- Feature timing, latency stability, held-out query evaluation, and online overhead remain central validity requirements.
- No claim is made that pruning adaptation, a learned controller, or RAG-level quality/latency gains have been proven.

## Archive map

- [`experiments/adaptive-retrieval/reports/`](../experiments/adaptive-retrieval/reports/): consolidated findings, failure analyses, incremental-reuse reports, controller and replay audits.
- [`experiments/adaptive-retrieval/code/`](../experiments/adaptive-retrieval/code/): selected user-authored experiment, analysis, profiling, and controller scripts. They are archival research code; paths, dependencies, and the underlying PLAID/ColBERT checkout may need to be configured before rerunning.
- [`experiments/system-benchmarks/`](../experiments/system-benchmarks/): PLAID/WARP/IGP/GIGP+ study notes and evaluation protocol.
- [`experiments/multivector-benchmarks/`](../experiments/multivector-benchmarks/): aggregate benchmark tables, comparison notes, and figures.
- [`experiments/adaptive-retrieval/results/`](../experiments/adaptive-retrieval/results/): aggregate policy, feature-stability, pruning-risk tables, and plots.
- [`papers/notes/`](../papers/notes/): related-work interpretation and reproduction-status notes.
- [`presentations/`](../presentations/): an earlier project presentation, retained as a dated historical artifact rather than a statement of current status.

## Public-repository exclusions

This is a curated research archive, not a byte-for-byte upload of the workstation or server. It excludes credentials, server addresses/private filesystem paths, raw datasets and query text, vector indexes, large per-query candidate/profile dumps, model weights, caches, temporary files, third-party source trees, and large execution logs. Aggregate artifacts are included only where their measurement scope and limitations are documented.
