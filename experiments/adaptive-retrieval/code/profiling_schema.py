"""Schema helpers for per-query multi-budget profiling."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

BUDGETS = (400, 512, 768, 1024)


@dataclass
class Metrics:
    recall_at_10: Optional[float] = None
    recall_at_100: Optional[float] = None
    mrr: Optional[float] = None
    ndcg: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {"recall@10": self.recall_at_10, "recall@100": self.recall_at_100,
                "mrr": self.mrr, "ndcg": self.ndcg}


@dataclass
class SearchState:
    candidate_overlap: Optional[float] = None
    cutoff_margin: Optional[float] = None
    tail_slope: Optional[float] = None
    topk_churn: Optional[float] = None
    score_entropy: Optional[float] = None
    score_gain: Optional[float] = None
    new_candidate_fraction: Optional[float] = None


@dataclass
class CandidateInfo:
    candidate_count: Optional[int] = None
    candidate_generation_failure: Optional[bool] = None
    candidate_ids: Optional[list[int]] = None
    topk_ids: Optional[list[int]] = None


@dataclass
class BudgetProfile:
    latency_ms: Optional[float] = None
    metrics: Metrics = field(default_factory=Metrics)
    search_state: SearchState = field(default_factory=SearchState)
    candidate_info: CandidateInfo = field(default_factory=CandidateInfo)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


@dataclass
class QueryBudgetProfile:
    query_id: str
    query_text: str = ""
    budgets: Dict[str, BudgetProfile] = field(default_factory=dict)
    transition_features: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    def ensure_budgets(self, budgets: Iterable[int] = BUDGETS) -> None:
        for budget in budgets:
            self.budgets.setdefault(str(int(budget)), BudgetProfile())

    def to_dict(self) -> Dict[str, Any]:
        self.ensure_budgets()
        return {"query_id": self.query_id, "query_text": self.query_text,
                "budgets": {key: value.to_dict() for key, value in sorted(
                    self.budgets.items(), key=lambda item: int(item[0]))},
                "transition_features": self.transition_features}


def profile_from_dict(data: Mapping[str, Any]) -> QueryBudgetProfile:
    profile = QueryBudgetProfile(str(data.get("query_id", "")), str(data.get("query_text", "")),
                                 transition_features=dict(data.get("transition_features", {}) or {}))
    for key, raw in (data.get("budgets", {}) or {}).items():
        raw, metrics = raw or {}, (raw or {}).get("metrics", {}) or {}
        state, candidate = (raw or {}).get("search_state", {}) or {}, (raw or {}).get("candidate_info", {}) or {}
        profile.budgets[str(key)] = BudgetProfile(
            latency_ms=raw.get("latency_ms"),
            metrics=Metrics(metrics.get("recall@10"), metrics.get("recall@100"), metrics.get("mrr"), metrics.get("ndcg")),
            search_state=SearchState(**{name: state.get(name) for name in SearchState.__dataclass_fields__}),
            candidate_info=CandidateInfo(candidate.get("candidate_count"), candidate.get("candidate_generation_failure"),
                                         candidate.get("candidate_ids"), candidate.get("topk_ids")),
        )
    profile.ensure_budgets()
    return profile
