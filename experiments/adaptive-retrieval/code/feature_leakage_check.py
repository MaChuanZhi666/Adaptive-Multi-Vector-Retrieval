"""Document which profile fields are available before a budget decision."""
from __future__ import annotations

from pathlib import Path


DECISION_TIME = [
    "query_length", "entity_count", "candidate_count", "candidate_generation_failure",
    "candidate_overlap", "cutoff_margin", "tail_slope", "topk_churn",
    "score_entropy", "score_gain", "new_candidate_fraction", "truncation_risk",
]
POST_RETRIEVAL = [
    "final recall@10", "final recall@100", "mrr", "ndcg", "oracle_budget",
    "answer quality", "answer F1", "groundedness", "1024 final result",
]


def main() -> None:
    out = Path("feature_availability.md")
    lines = ["# Controller feature availability", "", "| Feature | Available before decision? | Notes |", "|---|---|---|"]
    for name in DECISION_TIME:
        lines.append(f"| `{name}` | YES | Can be observed before final exact retrieval |")
    for name in POST_RETRIEVAL:
        lines.append(f"| `{name}` | NO | Evaluation/label only; forbidden as controller input |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
