"""Dependency-free feature extraction helpers for profiling."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence


def topk_churn(old_pids: Optional[Iterable[Any]], new_pids: Optional[Iterable[Any]], k: int = 100) -> Optional[float]:
    """Return 1 - intersection/K; None when either TopK is unavailable."""
    if old_pids is None or new_pids is None:
        return None
    old, new = list(old_pids)[:k], list(new_pids)[:k]
    denom = min(k, len(old), len(new))
    if denom <= 0:
        return None
    return 1.0 - len(set(old) & set(new)) / float(denom)


def candidate_overlap(previous: Optional[Iterable[Any]], current: Optional[Iterable[Any]]) -> Optional[float]:
    """Candidate-set overlap; intentionally distinct from TopK churn."""
    if previous is None or current is None:
        return None
    previous_set, current_set = set(previous), set(current)
    denom = min(len(previous_set), len(current_set))
    return None if denom <= 0 else len(previous_set & current_set) / float(denom)


def extract_search_state(raw: Mapping[str, Any], previous_topk: Optional[Sequence[Any]] = None,
                         current_topk: Optional[Sequence[Any]] = None) -> dict:
    """Map existing runtime signal names; unavailable values remain None."""
    return {
        "candidate_overlap": raw.get("candidate_overlap_previous"),
        "cutoff_margin": raw.get("approx_cutoff_margin"),
        "tail_slope": raw.get("tail_slope"),
        "topk_churn": topk_churn(previous_topk, current_topk),
        "score_entropy": raw.get("approx_score_entropy"),
        "score_gain": raw.get("approx_score_gain_vs_previous_budget"),
        "new_candidate_fraction": raw.get("stage2_new_candidate_fraction", raw.get("newly_admitted_fraction")),
    }
