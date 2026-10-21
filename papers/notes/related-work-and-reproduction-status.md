# Related work: connection to the PLAID research

This note records the project's interpretation and reproduction status for LAET, DARTH, and OMEGA. The papers are conceptual references for adaptive search, not components of PLAID and not completed reproductions in this project.

| Work | Main idea (project-level summary) | Relation to this project | Reproduction status |
|---|---|---|---|
| LAET (SIGMOD 2020) | Uses intermediate search information to predict whether more ANN search is worthwhile and stop early. | Motivates using search-state signals to decide whether another retrieval budget increment is useful. | Original benchmark/index setup was not completed; no paper-level result reproduction claimed. |
| DARTH (SIGMOD) | Targets a requested recall and adapts when/how often to check search progress. | Motivates target-quality-aware budget decisions and adaptive checking frequency. | Original system/data requirements were not met; only code/dependency feasibility and idea transfer were studied. |
| OMEGA (arXiv:2603.06159; described in project notes as SIGMOD '27) | Reuses a learned search capability across multiple K values and predicts remaining search benefit. | Motivates reusing work across retrieval requests/budget stages rather than restarting search. | No full reproduction on the paper's original benchmark; treat as related-work study only. |

## Boundary of the analogy

These methods concern ANN search algorithms and benchmark configurations that differ from PLAID's multi-vector retrieval pipeline. In this project, `ndocs` is a candidate/search budget and PLAID has multiple interacting stages, including candidate generation, pruning/filtering, decompression, and late-interaction scoring. A PLAID two-stage approximate→exact experiment is not equivalent to the original LAET, DARTH, or OMEGA algorithm.

The research contribution being explored is whether online-available signals can allocate work across PLAID stages while preserving retrieval/RAG evidence quality. It remains a hypothesis until tested with actual stage-boundary replay and independent evaluation.

Official references:

- [LAET DOI](https://dl.acm.org/doi/10.1145/3318464.3380600)
- [DARTH DOI](https://dl.acm.org/doi/10.1145/3749160)
- [OMEGA preprint](https://arxiv.org/html/2603.06159)
